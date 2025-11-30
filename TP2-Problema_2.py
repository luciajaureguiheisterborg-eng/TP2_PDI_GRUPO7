import cv2
import matplotlib.pyplot as plt

def img_procesada(img_gris):
    """
    Realiza un preprocesamiento morfológico.
    Parámetros: img_gray: Imagen en escala de grises.
    Retorna:
            - img_resaltada: Imagen luego del filtrado Top-Hat.
            - img_bin: Imagen binarizada.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7)) 
    img_resaltada = cv2.morphologyEx(img_gris, cv2.MORPH_TOPHAT, kernel)

    img_resaltada_norm = cv2.normalize(img_resaltada, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 

    _, img_bin = cv2.threshold(img_resaltada_norm, 100, 255, cv2.THRESH_BINARY)
    return img_resaltada, img_bin

def filtrar_componentes(m_estadistica):
    """
    Filtra componentes conectadas para seleccionar posibles letras basandose en alineación y distancia.
    Parámetros: m_estadistica: Estadísticas de componentes conectadas obtenidas mediante connectedComponentsWithStats. 
    Cada entrada contiene: [x, y, w, h, area]
    Retorna: Lista de componentes filtradas que cumplen criterios de razón de aspecto, área y alineación horizontal.
    """
    aspect_min = 1.3
    aspect_max = 5.0
    area_min = 15
    area_max = 150

    candidatos = []
    for st in m_estadistica:
        x = st[0]
        y = st[1]
        w = st[2]
        h = st[3]
        area = st[4]

        aspect_ratio = h / w  
        if (aspect_min <= aspect_ratio <= aspect_max) and (area_min <= area <= area_max):
            candidatos.append(st) 

    caracteres = []
    for i in range(len(candidatos)):
        for j in range(len(candidatos)):
            if i == j:
                continue

            x1, y1, w1, h1, a1 = candidatos[i]
            x2, y2, w2, h2, a2 = candidatos[j]

            dist_x = abs(x1 - x2) 
            dist_y = abs(y1 - y2) 
            
            if dist_x <= 20 and dist_y <= 5:
                caracteres.append(candidatos[i])
                caracteres.append(candidatos[j])

    letras_unicas = []
    vistos = set()
    for st in caracteres:
        t = tuple(st)
        if t not in vistos:
            vistos.add(t)
            letras_unicas.append(st)

    return letras_unicas

def detector_objetos(img_bin):
    """
    Detecta componentes conectadas en una imagen binaria y aplica un filtrado para identificar aquellas que podrían ser caracteres.
    Parámetros: Imagen binaria resultante del preprocesamiento.
    Retorna: Lista de componentes conectadas filtradas.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, 8, cv2.CV_32S)

    letras_det = filtrar_componentes(stats)
    return letras_det

def recortar_letras(img_gris, img_resaltada, letras_det):
    """
    Dibuja rectángulos sobre las letras detectadas y muestra recortes individuales
    de cada componente reconocida como carácter.
    Parámetros:
        img_gris: Imagen original en escala de grises.
        img_resaltada: Imagen filtrada mediante Top-Hat.
        letras_det: Lista de componentes conectadas filtradas.
    Retorna: Muestra:
            - Imagen de caracteres detectados.
            - Recortes individuales de cada letra ordenada.
    """
    img_color = cv2.cvtColor(img_gris.copy(), cv2.COLOR_GRAY2BGR)

    for st in letras_det:
        x, y, w, h, area = st
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 1)

    plt.figure(figsize=(12, 5))
    plt.imshow(img_color)
    plt.title("Caracteres detectadas")
    plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()

    if len(letras_det) > 0:
        
        letras_ordenadas = sorted(letras_det, key=lambda st: st[0])

        num_letras = len(letras_ordenadas)
        plt.figure(figsize=(2 * num_letras, 3))

        for i, st in enumerate(letras_ordenadas, start=1):
            x, y, w, h, area = st
            recorte = img_resaltada[y:y + h, x:x + w]

            plt.subplot(1, num_letras, i)
            plt.imshow(recorte, cmap='gray')
            plt.title(f"Caracter nro {i}")
            plt.xticks([]), plt.yticks([])

        plt.tight_layout()
        plt.show()
    else:
        print("No se detectaron letras")

def ejecutar():
    if True:
        for nro in range(1,13):
            if nro <= 9:
                img=cv2.imread(f"img0{nro}.png")
            else:
                img=cv2.imread(f"img{nro}.png")
            print(f"Procesando: {nro}")
 
            img_gris= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resaltada, img_thresh = img_procesada(img_gris)
     
            letras_det = detector_objetos(img_thresh)

            recortar_letras(img_gris, img_resaltada, letras_det)
if True:
    ejecutar()