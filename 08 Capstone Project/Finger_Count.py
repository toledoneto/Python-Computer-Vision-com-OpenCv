import cv2
import numpy as np
from sklearn.metrics import pairwise

#############################################################
# Variáveis globais

# fundo da img (background - bg) que será atualizado conforme necessário
background = None

# peso acumulado
accumulated_weight = 0.5

# escolha manual do ROI para se colocar a mão a fim de detectar
# no vídeo, será representado como um retângulo vermelho onde a mão deverá ficar
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600


#############################################################
# Valor médio do fundo da img
def calc_accum_avg(frame, accumulated_weight):
    '''
    Given a frame and a previous accumulated weight, computed the weighted average of the image passed in.
    '''
    
    # pega o fundo da img
    global background

    # primeira vez: cria o bg fazendo uma cópia do frame
    if background is None:
        background = frame.copy().astype("float")
        return None

    # computa o peso, acumula e atualiza o bg: acumula o peso de acordo com a media móvel
    cv2.accumulateWeighted(frame, background, accumulated_weight)

    # essa função não retorna nada se tiver bg, apenas modifica o valor da var global background


#############################################################
# segmenta a mão
def segment(frame, threshold=25):
    global background
    
    # calcula a diferença do bg original com o atual em que a mão está presente
    diff = cv2.absdiff(background.astype("uint8"), frame)

    # aplicando um th para pergarmos o plano de cima (foreground - fg)
    # como o primeiro retorno é desnecessário, descartamos usando um _
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Pega o contorno externo da img, apenas o contorno é necessário
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # se o contorno for 0, não pegamos nenhum contorno
    if len(contours) == 0:
        return None
    else:
        # pelo jeito que foi feito o algoritmo, a maior área externa é o contorno da mão
        hand_segment = max(contours, key=cv2.contourArea)
        
        # retorno da mão e da img após o th
        return thresholded, hand_segment


#############################################################
# Contando os dedos com Convex Hull
# Convex Hull: desenha um polígono nos pontos mais externos de um frame
def count_fingers(thresholded, hand_segment):

    # fazendo o Convex Hull
    conv_hull = cv2.convexHull(hand_segment)

    # Pegando os pontos mais no topo, em baixo, esq e dir
    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    # assumindo que o "meio" da mão está no cruzamento das linhas topo-baixo + esq-dir
    # esses serão os ptos centrais. Dois / força o retorno de um int
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2
    
    # calculando a > dist entre o centro da mão e os pontos mais extremos
    # provavelmente será o seu polegar
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    
    # pega a > dist
    max_distance = distance.max()
    
    # cria um círculo com 80% da > dist
    # qualquer objeto que estiver próx à borda do círculo, até msm fora, provavelmente será um dedo levantado
    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)

    # pega um ROI que engloba apenas esse círculo
    # shape[:2] para pegarmos apenas o (x, y), os canais de cores não interessam
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    
    # desenha o ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)

    # fazendo um bit-wise AND usando o ROI cricular anterior como mask
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # contornos do ROI
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # início da contagem: 0
    count = 0

    # tentativa de contar os dedos
    for cnt in contours:
        
        # Bounding box dos contornos
        (x, y, w, h) = cv2.boundingRect(cnt)

        # para contarmos os dedos, temos duas condições
        # 1. região de contorno não é o pulso
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        
        # 2. num de pontos não está mais de 25% fora do ROI circular, ou seja, são obj fora da mão
        limit_points = ((circumference * 0.25) > cnt.shape[0])

        # atualizando a contagem
        if out_of_wrist and limit_points:
            count += 1

    return count


#############################################################
# Executando o programa
cam = cv2.VideoCapture(0)

# Inicia a contagem
num_frames = 0

while True:
    # recebe o frame atual
    ret, frame = cam.read()

    # vira o frame para evitar o "efeito espelhado"
    frame = cv2.flip(frame, 1)

    # copia o frame
    frame_copy = frame.copy()

    # pega o ROI do frame
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    # aplica escala de cinza e blur no ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Usando os primeiros 60 frames que calculam a média do bg
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Finger Count", frame_copy)
            
    else:
        # segmentando a mão
        hand = segment(gray)

        # checando se foi possível encontrar a mão
        if hand is not None:
            
            # unpack
            thresholded, hand_segment = hand

            # Desenhando contorno ao redor da mão
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 1)

            # contando os dedos
            fingers = count_fingers(thresholded, hand_segment)

            # mostrando a contagem
            cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # mostrando a img pós th
            cv2.imshow("Thesholded", thresholded)

    # desenhando o ROI retangular na cópia da img
    # Draw ROI Rectangle on frame copy
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)

    # incrementa o num de frames
    num_frames += 1

    # mostra o frame com a mão segmentada
    cv2.imshow("Finger Count", frame_copy)

    # fechar a janela
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# libera cam e destrói janelas
cam.release()
cv2.destroyAllWindows()
