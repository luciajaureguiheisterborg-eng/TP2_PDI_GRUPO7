import cv2
import numpy as np

def preprocesar_imagen(imagen):
    """
    Convierte la imagen a escala de grises y aplica un suavizado Gaussiano.
    parametro: imagen (np.ndarray): Imagen en formato BGR.
    Retorna: np.ndarray: Imagen suavizada en escala de grises.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) 
    return cv2.GaussianBlur(gris, (3, 3), 0)

def detectar_monedas(imagen):
    """
    Detecta monedas en la imagen mediante la Transformada de Hough.
    Parámetros: imagen (np.ndarray): Imagen en escala de grises y suavizada.
    Retorna:np.ndarray | None: Arreglo con círculos detectados o None si no encuentra.
    """
    circulos_detectados = cv2.HoughCircles(imagen, cv2.HOUGH_GRADIENT, 1.2,90, param1=65, param2=170,minRadius=70, maxRadius=200)
    return circulos_detectados

def dibujar_circulos(imagen_original, circulos_detectados, imagen_final):
    """
    Dibuja los círculos detectados sobre la imagen final y crea una versión
    donde se enmascaran las monedas.
    Parámetros:
        imagen_original (np.ndarray): Imagen suavizada en grises.
        circulos_detectados (np.ndarray | None): Resultado de HoughCircles.
        imagen_final (np.ndarray): Imagen sobre la cual se dibujan los círculos.
    Retorna:np.ndarray, np.ndarray: (imagen_monedas, imagen_final)
    """
    imagen_monedas = imagen_original.copy()
    if circulos_detectados is not None:
        circulos_detectados = np.uint16(np.around(circulos_detectados))
        for punto in circulos_detectados[0, :]:
            a, b, r = punto[0], punto[1], punto[2]
           
            cv2.circle(imagen_final, (a, b), r, (255, 0, 0), 5)
            cv2.circle(imagen_monedas, (a, b), r, (0, 0, 0), -1)
  
    return imagen_monedas, imagen_final

def filtrar_componentes(estadisticas, etiquetas, centroides, area_minima=5):
    """
    Filtra componentes conectados por área mínima.
    Parámetros:
        estadisticas (np.ndarray): Stats de cada componente generados por connectedComponentsWithStats.
        etiquetas (np.ndarray): Imagen etiquetada.
        centroides (np.ndarray): Lista de centroides.
        area_minima (int): Área mínima para conservar un componente.
    Retorna: tuple:
            - nuevo número de etiquetas
            - nueva imagen etiquetada filtrada
            - estadísticas filtradas
            - centroides filtrados
    """
    estadisticas_filtradas = []
    indices_filtrados = []
    for i, stat in enumerate(estadisticas):
        _, _, _, _, area = stat
        if area >= area_minima:
            estadisticas_filtradas.append(stat)
            indices_filtrados.append(i)

    estadisticas_filtradas = np.array(estadisticas_filtradas)
    nuevas_etiquetas = np.zeros_like(etiquetas)
    for nueva_etiqueta, indice_antiguo in enumerate(indices_filtrados, start=1):
        nuevas_etiquetas[etiquetas == indice_antiguo] = nueva_etiqueta

    nuevos_centroides = centroides[indices_filtrados]
    nuevo_numero_etiquetas = len(indices_filtrados) + 1

    return nuevo_numero_etiquetas, nuevas_etiquetas, estadisticas_filtradas, nuevos_centroides

def clasificar_monedas(imagen, num_etiquetas, etiquetas, estadisticas, centroides):
    """
    Clasifica monedas según área estimada de su componente conectado.
    Parámetros:
        imagen (np.ndarray): Imagen original (solo por consistencia).
        num_etiquetas (int): Número total de etiquetas.
        etiquetas (np.ndarray): Imagen etiquetada.
        estadisticas (np.ndarray): Estadísticas por componente.
        centroides (np.ndarray): Centroides de cada componente.
    Retorna:tuple[int, int, int]: (cantidad $1, cantidad $0.50, cantidad $0.10)
    """
    cant_1 = 0
    cant_50 = 0
    cant_10 = 0
    for stat in estadisticas:
        area = stat[4]
        if area < 7000:
            cant_10 += 1
        elif 7000 <= area < 9000:
            cant_1 += 1
        elif 9000 <= area < 12000:
            cant_50 += 1
    return cant_1, cant_50, cant_10

def procesar_imagen_para_deteccion(imagen):
    """
    Preprocesa la imagen para facilitar la detección de dados: reduce tamaño, suaviza, modifica HSV, umbraliza,
                                                                 filtra por área y aplica apertura + clausura para limpiar la imagen.
    
    Parámetros:imagen (np.ndarray) 
    Retorna:imagen_apertura_clausura (np.ndarray)
    """
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    imagen_redimensionada = cv2.resize(imagen_hsv, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)  
    imagen_hsv = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2HSV)
    imagen_hsv = cv2.blur(imagen_hsv, (15, 15))
    imagen_modificada_hsv = imagen_hsv.copy()

    canal_h = imagen_hsv[:, :, 0] + 26
    canal_h[canal_h > 179] = 179
    canal_h[canal_h < 0] = 0 
    imagen_modificada_hsv[:, :, 0] = canal_h

    canal_s = cv2.blur(imagen_hsv[:, :, 1], (5, 5)) 
    canal_s[canal_s > 255] = 255 
    canal_s[canal_s < 0] = 0
    imagen_modificada_hsv[:, :, 1] = canal_s 
    canal_v = imagen_hsv[:, :, 2].copy()
    canal_v[canal_v > 255] = 255
    canal_v[canal_v < 0] = 0
    canal_v = cv2.blur(canal_v, (5, 5)) 
    imagen_modificada_hsv[:, :, 2] = canal_v

    imagen_bgr = cv2.cvtColor(imagen_modificada_hsv, cv2.COLOR_HSV2BGR) 
    imagen_gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY) 

    _, imagen_umbralizada = cv2.threshold(imagen_gris, 63, 255, cv2.THRESH_BINARY)
    num_etiquetas, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(imagen_umbralizada, connectivity=8)
    imagen_filtrada = np.zeros_like(imagen_umbralizada, dtype=np.uint8)
    for i in range(1, num_etiquetas): 
        
        if estadisticas[i, cv2.CC_STAT_AREA] >= 1300:  
            imagen_filtrada[etiquetas == i] = 255   

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
    imagen_cerrada = cv2.morphologyEx(imagen_filtrada.copy(), cv2.MORPH_CLOSE, kernel, iterations=9) 
    imagen_apertura_clausura = cv2.morphologyEx(imagen_cerrada.copy(), cv2.MORPH_OPEN, kernel, iterations=9) 

    return imagen_apertura_clausura

def detectar_figuras_por_factor_forma(imagen, fp_min=0.063, fp_max=0.073):
    """
    Detecta figuras aproximadas a cuadrados utilizando el factor de forma.
    Parámetros:
        imagen (np.ndarray): Imagen binaria o filtrada.
        fp_min (float): Factor de forma mínimo aceptado.
        fp_max (float): Factor de forma máximo aceptado.
    Retorna: list[np.ndarray]: Lista de contornos que corresponden a cuadrados detectados.
    """
    imagen_dibujada = imagen.copy()
    bordes = cv2.Canny(imagen_dibujada, 50, 150)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cuadrados = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if perimetro == 0:
            continue
        fp = area / (perimetro ** 2)
        if not (fp_min < fp < fp_max):
            cuadrados.append(contorno)
    return cuadrados

def contar_circulos(dados, imagen_final):
    """
    Dado cada contorno de un dado detectado, recorta la región y cuenta los puntos
    mediante HoughCircles.
    Parámetros:
        dados (list[np.ndarray]): Lista de contornos detectados como dados.
        imagen_final (np.ndarray): Imagen donde se dibujan resultados.
    Retorna: tuple:
            - dict con el valor de cada dado
            - puntaje total
            - imagen resultante anotada
    """
    resultados_dados = {}
    puntaje_total = 0
    numero_dado = 1
    imagen_original = imagen_final.copy()

    for contorno in dados:
        x, y, w, h = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        if area < 500:
            continue

        region = imagen_original[y:y+h, x:x+w]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 3)

        circulos = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.3, minDist=10,param1=10, param2=25, minRadius=1, maxRadius=30)

        puntaje = 0
        if circulos is not None:
            circulos = np.round(circulos[0, :]).astype("int")
            for (cx, cy, r) in circulos:
                puntaje += 1
                cv2.circle(region, (cx, cy), r, (0, 255, 0), 1)

        resultados_dados[f"Dado {numero_dado}"] = puntaje
        puntaje_total += puntaje
        numero_dado += 1

        texto_x = x + w + 5
        texto_y = y + 20

        imagen_final = cv2.putText(imagen_final, f"Puntaje: {puntaje_total}",(texto_x, texto_y), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)
        imagen_final = cv2.rectangle( imagen_final, (x, y), (x+w, y+h), (0, 0, 255), 2)

    print(resultados_dados)
    return resultados_dados, puntaje_total, imagen_final

def ejecutar(imagen_final):
    """
    Ejecuta todas las etapas:
      1. Preprocesamiento y detección de monedas
      2. Binarización y componentes conectados
      3. Filtrado por área y clasificación de monedas
      4. Procesamiento HSV para destacar dados
      5. Detección de dados mediante factor de forma
      6. Conteo de puntos de cada dado 
    Parámetros: imagen_final (np.ndarray): Imagen donde se dibujan los resultados finales.
    Retorna: np.ndarray: Imagen final
    """
    imagen = cv2.imread("monedas.jpg")
    imagen_suavizada = preprocesar_imagen(imagen)
    circulos = detectar_monedas(imagen_suavizada)
    imagen_con_circulos, imagen_final = dibujar_circulos(imagen_suavizada, circulos, imagen_final)
    _, binaria = cv2.threshold(imagen_con_circulos, 1, 255, cv2.THRESH_BINARY_INV)
    imagen_redim = cv2.resize(binaria, None, fx=0.5, fy=0.5)
    imagen_final = cv2.resize(imagen_final, None, fx=0.5,fy=0.5)

    _, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(imagen_redim, 8)
    _, etiquetas_filtradas, estad_filtradas, centroides_filtrados = filtrar_componentes(stats, etiquetas, centroides)

    c1, c50, c10 = clasificar_monedas(imagen, _, etiquetas_filtradas, estad_filtradas, centroides_filtrados)
    print("Cantidad $1:", c1)
    print("Cantidad $0.50:", c50)
    print("Cantidad $0.10:", c10)

    imagen_final = cv2.resize(imagen_final, (imagen.shape[1], imagen.shape[0]))
    imagen_final = cv2.resize(imagen_final, None, fx=0.5, fy=0.5)

    imagen_proc = procesar_imagen_para_deteccion(cv2.imread("monedas.jpg"))
    cuadrados = detectar_figuras_por_factor_forma(imagen_proc)

    dados, puntaje, imagen_final = contar_circulos(cuadrados, imagen_final)

    print("Cantidad de dados:", len(dados))
    print("Suma total:", puntaje)
    return imagen_final

if True:
    imagen_principal = cv2.imread("monedas.jpg")
    imagen_final_procesada = ejecutar(imagen_principal)
    
    cv2.imshow("Imagen Final", cv2.resize(imagen_final_procesada,(imagen_final_procesada.shape[1] // 3, imagen_final_procesada.shape[0] // 3)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()