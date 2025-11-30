import cv2
import numpy as np
#---------------------------------------------------------------------------------------------------------------------------------------
#    PREPROCESA LA IMAGEN PASA A GRIS Y SUAVIZA
#---------------------------------------------------------------------------------------------------------------------------------------
def preprocesar_imagen(imagen): # #Sirve para preparar la imagen antes de detectar monedas.
    """
    Convierte la imagen a escala de grises y aplica un suavizado Gaussiano.
    parametro: imagen (np.ndarray): Imagen en formato BGR.
    Retorna: np.ndarray: Imagen suavizada en escala de grises.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) # para convertir la imagen de color (BGR) a escala de grises.
#Cada pixel BGR (azul, verde, rojo) como:
#(b, g, r)
#se convierte en un solo valor de intensidad (0-255).
    return cv2.GaussianBlur(gris, (3, 3), 0) #Suaviza bordes Y Mejora que Hough detecte círculos sin confundir ruido con monedas
#El objetivo es quitar ruido, estandarizar la imagen y dejarla lista para usar en HoughCircles.
#Ese tercer parámetro es la desviación estándar sigma del filtro gaussiano.Cuando ponés 0, OpenCV calcula sigma automáticamente según el tamaño del kernel.

#-----------------------------------------------------------------------------------------------------------------------------------------
# DETECCION DE MONEDAS MEDIANTE LA TRASFORMADA DE HOUGH
#-----------------------------------------------------------------------------------------------------------------------------------------

def detectar_monedas(imagen):
    """
    Detecta monedas en la imagen mediante la Transformada de Hough.
    Parámetros: imagen (np.ndarray): Imagen en escala de grises y suavizada.
    Retorna:np.ndarray | None: Arreglo con círculos detectados o None si no encuentra.
    """
    circulos_detectados = cv2.HoughCircles(imagen, cv2.HOUGH_GRADIENT, 1.2,90, param1=65, param2=170,minRadius=70, maxRadius=200)
    return circulos_detectados
"""
✔ dp = 1.2-->Factor de reducción/resolución.Hace que la detección sea más robusta pero más liviana.
✔ minDist = 90 --> Distancia mínima entre dos círculos detectados.Evita detectar la misma moneda varias veces.
✔ param1 = 65 --> Umbral para Canny interno.Controla qué tan sensibles son los bordes.
✔ param2 = 170 --> Umbral de acumulador (sensibilidad de detección).
Más alto → detecta menos círculos pero más seguros
Más bajo → detecta más, pero arriesga falsos positivos
✔ minRadius = 70 --> Tamaño mínimo de moneda.
✔ maxRadius = 200 -->Tamaño máximo de moneda
"""
#------------------------------------------------------------------------------------------------------------------------------------------
# DIBUJAR CIRCULOS (MONEDAS) Y ENMASCARAR
#------------------------------------------------------------------------------------------------------------------------------------------
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
    """
    La función tiene dos objetivos: Dibujar los círculos detectados (las monedas) sobre una imagen final para visualizarlos. Crear una imagen donde las monedas están enmascaradas (pintadas de negro), útil para segmentación o análisis.
    """
    imagen_monedas = imagen_original.copy() #Copia de la imagen original para crear la versión con monedas tapadas
    if circulos_detectados is not None: # si hay circulos
        circulos_detectados = np.uint16(np.around(circulos_detectados))  #Redondear y convertir los círculos a enteros, como HoughCircles devuelve valores float.Se redondean (around) porque los píxeles deben ser enteros.
#uint16 evita valores negativos.
        for punto in circulos_detectados[0, :]: #recorre la lista de cada circulo detectado 
            # cada punto de esa lista es:
            # a = centro x
            # b = centro y
            # r = radio
            a, b, r = punto[0], punto[1], punto[2] 
           
            cv2.circle(imagen_final, (a, b), r, (255, 0, 0), 5) #Se dibuja el borde del círculo en rojo (BGR) sobre la imagen final.Grosor 5 px.
            cv2.circle(imagen_monedas, (a, b), r, (0, 0, 0), -1) #Crea un círculo relleno (-1) color negro. Esto tapa la moneda en la imagen.
  
    return imagen_monedas, imagen_final

#-----------------------------------------------------------------------------------------------------------------------------------------
# FILTRAR COMPONENTES SEGUN EL AREA MINIMA 
#--------------------------------------------------------------------------------------------------------------------------------------------

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
    """
✔️ quedarte solo con los componentes que te sirven
✔️ descartar los que tienen área muy chica
✔️ volver a generar una imagen etiquetada limpia
✔️ recalcular estadísticas y centroides solo para los componentes útiles
    """
    """
Cada stat contiene:
[x, y, width, height, area]

Si area >= area_minima → se conserva.

Se guardan:las estadísticas válidas y los índices originales (importantísimo para rearmar etiquetas)
    """
    estadisticas_filtradas = []
    indices_filtrados = [] #su índice original (importante para reetiquetar)
    for i, stat in enumerate(estadisticas):
        _, _, _, _, area = stat
        if area >= area_minima:
            estadisticas_filtradas.append(stat)
            indices_filtrados.append(i)

    estadisticas_filtradas = np.array(estadisticas_filtradas) #Convertir stats filtradas a array
    nuevas_etiquetas = np.zeros_like(etiquetas) #Crea una imagen con todas las etiquetas en 0 (fondo).
    """
etiquetas es una matriz del mismo tamaño que la imagen
Cada píxel tiene un número que indica a qué componente pertenece
    """
    for nueva_etiqueta, indice_antiguo in enumerate(indices_filtrados, start=1): #Re-etiqueta los componentes filtrados #indice antiguo: es la etiqueta original en la imagen 
        #start=1 --> saltándose la etiqueta 0, porque la etiqueta 0 siempre es el fondo.
        nuevas_etiquetas[etiquetas == indice_antiguo] = nueva_etiqueta

    nuevos_centroides = centroides[indices_filtrados] #Solo conserva los centroides de los componentes aceptados.
    nuevo_numero_etiquetas = len(indices_filtrados) + 1 #Siempre suma 1 porque: etiqueta 0 = fondo
    """
En una imagen etiquetada, la etiqueta 0 siempre representa el fondo.
Las etiquetas de los componentes reales comienzan desde 1.
Por lo tanto, si tenemos N componentes válidos, la cantidad total de etiquetas será N + 1.
    """
    """
    | Acción                            | ¿Para qué sirve?                                                         |
    | --------------------------------- | ------------------------------------------------------------------------ |
    | **Reetiquetar con start=1**       | Para asignar nuevas etiquetas consecutivas empezando desde 1 (fondo = 0) |
    | **indices_filtrados**             | Son las etiquetas que sobrevivieron al filtro                            |
    | **nuevas_etiquetas[...] = ...**   | Cambia cada etiqueta vieja por una nueva                                 |
    | **centroides[indices_filtrados]** | Se quedan solo los centroides válidos                                    |
    | **len(indices_filtrados) + 1**    | Cuenta las etiquetas totales (fondo + componentes válidos)               |

    """
    return nuevo_numero_etiquetas, nuevas_etiquetas, estadisticas_filtradas, nuevos_centroides #Se retorna todo lo actualizado
#cantidad de etiquetas (incluyendo fondo)
#nueva imagen etiquetada limpia
#estadisticas filtradas
#centroides filtrados

#----------------------------------------------------------------------------------------------------------------------------------------
# CLASIFICA CADA MONEDA S/ SU AREA
#------------------------------------------------------------------------------------------------------------------------------------------
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
    print(f" la cantidad de monedas es{len(estadisticas)}")
    for stat in estadisticas:
        """
        stat es:[x, y, width, height, area]
        """
        area = stat[4] # el area 
        print(f"el area de la moneda es {area}")
        if area < 7000:
            cant_10 += 1
        elif 7000 <= area < 9000:
            cant_1 += 1
        elif 9000 <= area < 12000:
            cant_50 += 1
    return cant_1, cant_50, cant_10
#--------------------------------------------------------------------------------------------------------------------------------------------
# PREPOCESA LA IMAGEN PARA LA DETECCION DE DADOS
#--------------------------------------------------------------------------------------------------------------------------------------------
def procesar_imagen_para_deteccion(imagen):
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    """
HSV separa color (H), saturación (S) e iluminación (V).
Esto permite manipular el color mucho mejor que en BGR.
    """
#conversión de BGR a HSV,usamos HSV xq podemos trabajar con el canal V para controlar el brillo sin afectar el color.Eso hace más fácil separar monedas y dados del fondo
    imagen_redimensionada = cv2.resize(imagen_hsv, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) 
     #Reduce la imagen al 50 % del tamaño original, con el metodo de interpolacionlineal, 
    """
         Redimensionar la imagen a la mitad ¿Para qué?
        Acelera procesamiento.
        Reduce ruido.
        Reduce la cantidad de componentes conectados falsos.
     """
    imagen_hsv = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2HSV) #Convertir nuevamente
    imagen_hsv = cv2.blur(imagen_hsv, (15, 15))  #suavizar
    """
    Vuelve a pasar a HSV (por el resize). Aplica un blur fuerte 15×15, que:elimina ruido, suaviza bordes,hace más homogéneo el colorfavorece un umbralizado limpio

    """
    imagen_modificada_hsv = imagen_hsv.copy() #Crea una copia para modificar canales sin alterar la original en memoria

    canal_h = imagen_hsv[:, :, 0] + 26 
    """
    ¿Qué es Hue (Matiz)?
El canal H en HSV representa el color puro:
0–10 → rojos
30–60 → amarillos
60–100 → verdes
100–140 → cyan
140–179 → azul/violeta
➕ ¿Por qué sumar 26?
👉 Para cambiar el color de toda la imagen, desplazándolo en el círculo de colores.
Esto se usa mucho para:

resaltar ciertos colores

separar mejor objetos del fondo

mejorar el contraste entre zonas

facilitar la segmentación
En tu caso:
Probablemente están intentando desplazar el color del dado o del fondo para que la umbralización salga más clara.
    """
    #Se hace para realzar la diferencia entre tonos de las monedas/dados y el fondo
    canal_h[canal_h > 179] = 179 #evita salirse del rango [0,179].
    canal_h[canal_h < 0] = 0 #Como el canal H solo puede tener valores entre 0 y 179,esta parte asegura que si sumamos 26 y nos pasamos de 179, no se desborde.
    imagen_modificada_hsv[:, :, 0] = canal_h #tendrá un H desplazado.

    canal_s = cv2.blur(imagen_hsv[:, :, 1], (5, 5)) # #suaviza pequeñas variaciones de saturación
    #suaviza pequeñas variaciones de saturación
    canal_s[canal_s > 255] = 255 #fuerza a que los valores queden dentro del rango válido [0,255].
    canal_s[canal_s < 0] = 0
    imagen_modificada_hsv[:, :, 1] = canal_s #Reemplaza el canal S en la copia modificada por la versión suavizada.

    canal_v = imagen_hsv[:, :, 2].copy() #copy() sirve para trabajar de forma segura en un canal sin romper la imagen original.
    canal_v[canal_v > 255] = 255
    canal_v[canal_v < 0] = 0
    canal_v = cv2.blur(canal_v, (5, 5)) #SUAVIZA: reduce pequeños reflejos puntuales y hace el brillo más homogéneo
    imagen_modificada_hsv[:, :, 2] = canal_v

    imagen_bgr = cv2.cvtColor(imagen_modificada_hsv, cv2.COLOR_HSV2BGR) #Convierte la versión HSV modificada de vuelta a BGR para operaciones que vienen después (y porque threshold se hará sobre escala de grises BGR→GRAY).
    #pasa devuelta a bgr
    imagen_gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY) #escala de grises, Pasa de BGR → gris
    #Por qué: es más simple y rápido umbralizar sobre un único canal de intensidad.

    _, imagen_umbralizada = cv2.threshold(imagen_gris, 63, 255, cv2.THRESH_BINARY)
    """
    Umbral fijo: todo píxel con intensidad > 63 → 255 (blanco), sino → 0 (negro).
    Resultado: imagen_umbralizada es binaria.
    Nota: el valor 63 fue elegido empíricamente; puede necesitar ajuste según iluminación.
    """
    num_etiquetas, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(imagen_umbralizada, connectivity=8) #Detecta componentes conectados en la binaria, considera 8-vecinos (diagonal incluida) para conexión.
    imagen_filtrada = np.zeros_like(imagen_umbralizada, dtype=np.uint8) #Crea una imagen negra (ceros) del mismo tamaño para llenar solo los componentes que pasan el filtro.
    
    for i in range(1, num_etiquetas): #recorre todos los componentes conectados detectados en la imagen
        #print(estadisticas[i,cv2.CC_STAT_AREA]) #PARA VER EL AREA
        if estadisticas[i, cv2.CC_STAT_AREA] >= 1300:  #si el area es mayor a 1300
            imagen_filtrada[etiquetas == i] = 255   
    """
    Recorre solo las etiquetas de objetos reales (empieza en 1 para ignorar fondo).
    Comprueba el área (estadisticas[i, cv2.CC_STAT_AREA]) y si es >= 1300 píxeles, copia ese componente a imagen_filtrada (lo pinta de blanco).
    Propósito: eliminar objetos muy pequeños (ruido), conservar solo regiones grandes (candidatas a dados).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #Crea un elemento estructurante (kernel) de tamaño 5×5 con forma elíptica
    imagen_cerrada = cv2.morphologyEx(imagen_filtrada.copy(), cv2.MORPH_CLOSE, kernel, iterations=9)
    """
    Aplica CLOSE (dilatación luego erosión), repetido 9 veces.
    Efecto: cierra huecos, une regiones cercanas, rellena pequeñas depresiones dentro de los objetos. Hace los dados más sólidos.
    """
    imagen_apertura_clausura = cv2.morphologyEx(imagen_cerrada.copy(), cv2.MORPH_OPEN, kernel, iterations=9) #OPEN = EROSIÓN seguida de DILATACIÓN.REPETIDO 9 VECES 
#cada iteración aplica la operación otra vez, amplificando el efecto.
    """
    Efecto: elimina pequeños ruidos remanentes, separa elementos muy finos adheridos a bordes, redondea/busca limpieza final.

    Nota: usar 9 iteraciones es una operación muy agresiva; funciona si tus objetos son grandes comparados con el kernel, pero puede erosionar objetos más pequeños (ajustar según imagen).
    """
    return imagen_apertura_clausura
#----------------------------------------------------------------------------------------------------------------------------------------
#  DETECTAR DADOS SEGUN EL FACTOR FORMA DE UN CUADRADO
#-----------------------------------------------------------------------------------------------------------------------------------------------
def detectar_figuras_por_factor_forma(imagen, fp_min=0.063, fp_max=0.073):
    """
    Detecta figuras aproximadas a cuadrados utilizando el factor de forma.
    Agrega el contorno SI su factor de forma NO está entre fp_min y fp_max.
    Parámetros:
        imagen (np.ndarray): Imagen binaria o filtrada.
        fp_min (float): Factor de forma mínimo aceptado.
        fp_max (float): Factor de forma máximo aceptado.
    Retorna: list[np.ndarray]: Lista de contornos que corresponden a cuadrados detectados.
    """
    imagen_dibujada = imagen.copy() #copia para no modificar la orig
    bordes = cv2.Canny(imagen_dibujada, 50, 150) #detecta bordes c canny 
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #encuentra contornos externos 

    cuadrados = [] #lista de cuadrados
    for contorno in contornos:
        area = cv2.contourArea(contorno) #calcula el area 
        perimetro = cv2.arcLength(contorno, True) #calcula el perimetro 
        if perimetro == 0: #evita divisiones por 0
            continue
        fp = area / (perimetro ** 2) #calcula el factor forma 
        if not(fp_min < fp < fp_max):
            cuadrados.append(contorno)
    return cuadrados #Una lista de contornos que representan cuadrados
#Esta función sirve para detectar figuras que tengan una forma parecida a un cuadrado, usando un índice matemático llamado factor de forma (FP).
#Sirve para detectar los DADOS en la imagen, porque los dados tienen forma cuadrada

#----------------------------------------------------------------------------------------------------------
# PARA CONTAR LOS CIRCULOS DENTRO DE CADA DADO 
#----------------------------------------------------------------------------------------------------------
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
    resultados_dados = {} #diccionario de cada dado con su valor
    puntaje_total = 0 #total del punttaje
    numero_dado = 1 #nro de dado 
    imagen_original = imagen_final.copy()
    
    for contorno in dados:
        #print(len(dados))
        x, y, w, h = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        print(area)
        """   
        0.0
        11618.0
        12637.5 se queda con estas dos
        3.5
        5.0
        """
        #obtiene el bounding box(rectangulo minimo q lo contiene)y el area
        if area < 500:
            continue

        region = imagen_original[y:y+h, x:x+w]
        #cv2.imshow("region dado",region)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 3)
        #pasa a gris y suavisa 

        circulos = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.3, minDist=10,param1=10, param2=25, minRadius=1, maxRadius=30)

        """
        dp = 1.3 → resolución para la acumulación.
        minDist = 10 → no detectes puntos pegados uno encima del otro.
        param1 / param2 → sensibilidad del detector.
        min-maxRadius → tamaño estimado de los puntos.
        """
       #detecta circulos de la region 
        puntaje = 0
        if circulos is not None:
            circulos = np.round(circulos[0, :]).astype("int") #redondea y pasa a enteros
            for (cx, cy, r) in circulos: #itera sobre cada circulo 
                puntaje += 1 #suma puntaje a medida q itera sobre los circulos
                cv2.circle(region, (cx, cy), r, (0, 255, 0), 1)
                #cv2.imshow("Circulo detectado", region)
        resultados_dados[f"Dado {numero_dado}"] = puntaje
        puntaje_total += puntaje
        numero_dado += 1

        texto_x = x + w + 5 #Dibuja en la imagen final
        texto_y = y + 20 #

        imagen_final = cv2.putText(imagen_final, f"Puntaje: {puntaje_total}",(texto_x, texto_y), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)
        imagen_final = cv2.rectangle( imagen_final, (x, y), (x+w, y+h), (0, 0, 255), 2)
#PARA PONER TEXTO Y VER LA IMAGEN 
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
    imagen = cv2.imread("monedas.jpg") #LEE LA IMAGEN
    imagen_suavizada = preprocesar_imagen(imagen) #SUAVIZA C LA 1° FUNCION 
    circulos = detectar_monedas(imagen_suavizada) #DETECTA CIRCULOS Q DEVUELVE [CX,CY,R] 
    imagen_con_circulos, imagen_final = dibujar_circulos(imagen_suavizada, circulos, imagen_final) #Pinta las monedas detectadas sobre la imagen.
    _, binaria = cv2.threshold(imagen_con_circulos, 1, 255, cv2.THRESH_BINARY_INV) #Binarizar para obtener componentes
    imagen_redim = cv2.resize(binaria, None, fx=0.5, fy=0.5) #REDIMENSIONA 
    imagen_final = cv2.resize(imagen_final,None, fx=0.5, fy=0.5) #REDIMENSIONA Ajusta tamaños para que connectedComponents funcione mejor.cv2.resize(binaria, None, fx=0.5, fy=0.5)

    _, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(imagen_redim, 8) #btener componentes conectados
    _, etiquetas_filtradas, estad_filtradas, centroides_filtrados = filtrar_componentes(stats, etiquetas, centroides) #Filtrar esos componentes 

    c1, c50, c10 = clasificar_monedas(imagen, _, etiquetas_filtradas, estad_filtradas, centroides_filtrados) #clasificar monedas con las nuevas etiquetas de coponentes 
    print("Cantidad $1:", c1)
    print("Cantidad $0.50:", c50)
    print("Cantidad $0.10:", c10)

    imagen_final = cv2.resize(imagen_final, (imagen.shape[1], imagen.shape[0]))
    imagen_final = cv2.resize(imagen_final, None, fx=0.5, fy=0.5)
#Ajustar tamaño de la imagen final
    imagen_proc = procesar_imagen_para_deteccion(cv2.imread("monedas.jpg")) #Preprocesar para detectar los dados
    #cv2.imshow("imagen preprocesada color",imagen_proc)
    cuadrados = detectar_figuras_por_factor_forma(imagen_proc) #Detectar contornos cuadrados (los dados)
#Usa factor de forma → encuentra los contornos cuadrados que son los dados.
    dados, puntaje, imagen_final = contar_circulos(cuadrados, imagen_final) #Contar los puntos (círculos) de cada dado

    print("Cantidad de dados:", len(dados))
    print("Suma total:", puntaje)
    return imagen_final

if True:
    imagen_principal = cv2.imread("monedas.jpg")
    imagen_final_procesada = ejecutar(imagen_principal)
    
    cv2.imshow("Imagen Final", cv2.resize(imagen_final_procesada,(imagen_final_procesada.shape[1] // 3, imagen_final_procesada.shape[0] // 3)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()