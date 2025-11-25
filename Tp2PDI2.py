import cv2
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

if True:
    for nro in range(9,11):
        if nro <= 9:
            imag=cv2.imread(f"img0{nro}.png")
        else:
            imag=cv2.imread(f"img{nro}.png")
        print(f"Procesando: {nro}")

        # 3) Paso a escala de grises
        img_gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        #imshow(img_gray)
        # 4) Aplico Top-Hat para resaltar regiones claras sobre fondo más oscuro
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))
        img_top_hat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, se)
        #imshow(img_top_hat)

        # 5) Normalizo la imagen Top-Hat a rango 0–255
        img_top_hat_norm = cv2.normalize(img_top_hat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #imshow(img_top_hat_norm)
        # 6) Umbralado binario (blanco/negro)
        _, img_thresh = cv2.threshold(img_top_hat_norm, 100, 255, cv2.THRESH_BINARY)
        #imshow(img_thresh)
        # 7) Componentes conectadas
        # stats: para cada componente -> [x, y, w, h, area]
        connectivity = 10
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_thresh, connectivity, cv2.CV_32S)

        # 8) Filtro las componentes que podrían ser letras según:
        #    - relación de aspecto (alto/ancho)
        #    - área
        aspect_min = 1.3 #Alto/Ancho ~  1.5-3.0
        aspect_max = 3.0
        area_min = 15
        area_max = 150

        candidatos = []  # posibles letras
        for st in stats:
            x = st[0]
            y = st[1]
            w = st[2]
            h = st[3]
            area = st[4]

            if w > 0:
                aspect_ratio = float(h) / float(w)
            else:
                aspect_ratio = 0

            if aspect_min <= aspect_ratio <= aspect_max and area_min <= area <= area_max:
                candidatos.append(st)

        # 9) De esos candidatos, busco los que estén cerca entre sí
        #    (alineados como letras de la patente)
        letras = []
        for i in range(len(candidatos)):
            for j in range(len(candidatos)):
                if i == j:
                    continue

                st_i = candidatos[i]
                st_j = candidatos[j]

                dx = abs(st_i[0] - st_j[0])  # diferencia en x
                dy = abs(st_i[1] - st_j[1])  # diferencia en y

                # Si están cerca horizontalmente y casi a la misma altura,
                # supongo que son letras vecinas
                if dx <= 20 and dy <= 5:
                    letras.append(st_i)
                    letras.append(st_j)

        # 10) Si no encontré letras, no puedo armar la patente
        if len(letras) == 0:
            print(f"No se detectó patente en la imagen {nro}")
            continue

        # 11) A partir de las letras detectadas, armo un rectángulo grande
        #     que encierra toda la patente
        x_min = min(st[0] for st in letras)
        y_min = min(st[1] for st in letras)
        x_max = max(st[0] + st[2] for st in letras)
        y_max = max(st[1] + st[3] for st in letras)

        # 12) Recorte de la patente (segmentación)
        patente_recorte = img_gray[y_min:y_max, x_min:x_max]

        # --- BINARIZACIÓN DE LA PATENTE SEGMENTADA ---

        # 1) Suavizado
        suave = cv2.GaussianBlur(patente_recorte, (5,5), 0)

        # 2) Realce de texto (tophat + blackhat)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        tophat = cv2.morphologyEx(suave, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(suave, cv2.MORPH_BLACKHAT, kernel)
        realzada = cv2.add(suave, tophat)
        realzada = cv2.subtract(realzada, blackhat)

        # 3) Otsu (el mejor umbral automático)
        _, bin_otsu = cv2.threshold(realzada, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4) Invertir (letras blancas, fondo negro)
        patente_bin = 255 - bin_otsu

        imshow(patente_bin)

        # 13) Dibujo el rectángulo sobre la imagen original (en escala de grises)
        img_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # 14) Muestro resultados
        plt.figure(figsize=(10, 4))
        plt.suptitle(f"Patente detectada - en la imagen {nro}")

        plt.subplot(1, 2, 1)
        plt.imshow(img_vis, cmap="gray")
        plt.title("Patente detectada")
        plt.xticks([]), plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.imshow(patente_recorte, cmap="gray")
        plt.title("Patente segmentada")
        plt.xticks([]), plt.yticks([])

        plt.tight_layout()
        plt.show(block=True)