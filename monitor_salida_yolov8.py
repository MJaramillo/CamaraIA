from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import time
from pymongo import MongoClient
import base64

# ======================
# Conexión a MongoDB
# ======================
uri = "mongodb://localhost:27017/"
client = MongoClient(uri)

# Verificar conexión
try:
    client.admin.command('ping')
    print("Conexión a MongoDB exitosa.")
except Exception as e:
    print("Error al conectar a MongoDB:", e)

db = client["AFORO"]
capturas_collection = db["CAPTURAS"]
exceso_aforo_collection = db["EXCESODEAFORO"]
historial_collection = db["HISTORIAL"]
zona_emergencia_collection = db["ZONADEEMERGENCIA"]

# ======================
# Variables globales
# ======================
p1 = (-1, -1)
p2 = (-1, -1)
area_seleccionada = False
imagen_copia = None
conteo_dentro = 0
conteo_fuera = 0
ultima_captura = 0
intervalo_captura = 30

# ======================
# Modelo YOLOv8
# ======================
model = YOLO("yolov8n.pt")  # Asegúrate de tener el modelo en el mismo directorio

# ======================
# Función para seleccionar área con mouse
# ======================
def dibujar_rectangulo(event, x, y, flags, param):
    global p1, p2, area_seleccionada

    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = (x, y)
        p2 = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        p2 = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        p2 = (x, y)
        area_seleccionada = True
        print(f"Área seleccionada: {p1} a {p2}")

# ======================
# Función para guardar imagen en base64
# ======================
def guardar_imagen_base64(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return img_base64

# ======================
# Guardar registros en MongoDB
# ======================
def guardar_registros_mongo(conteo_dentro, conteo_fuera, imagen, timestamp):
    imagen_base64 = guardar_imagen_base64(imagen)

    # Verificar que estamos recibiendo la imagen y los datos correctamente
    print("Guardando registro en MongoDB...")

    try:
        # Guardar captura
        capturas_collection.insert_one({
            "nombre_archivo": f"snapshot_{timestamp}.jpg",
            "timestamp": timestamp,
            "imagen_base64": imagen_base64
        })
        print(f"Captura guardada con nombre: snapshot_{timestamp}.jpg")

        # Guardar historial
        historial_collection.insert_one({
            "total_personas": conteo_dentro + conteo_fuera,
            "timestamp": timestamp
        })
        print(f"Historial guardado: {conteo_dentro + conteo_fuera} personas, {timestamp}")

        # Verificar aforo
        aforo_maximo = 5
        if conteo_dentro > aforo_maximo:
            exceso_aforo_collection.insert_one({
                "exceso_detectado": True,
                "personas_en_salida": conteo_dentro,
                "aforo_maximo": aforo_maximo,
                "timestamp": timestamp
            })
            print("Exceso de aforo detectado y registrado")

        # Guardar zona de emergencia
        zona_emergencia_collection.insert_one({
            "personas_en_salida": conteo_dentro,
            "timestamp": timestamp
        })
        print(f"Zona de emergencia registrada: {conteo_dentro} personas, {timestamp}")
    
    except Exception as e:
        print("Error al guardar en MongoDB:", e)

# ======================
# Contar personas
# ======================
def contar_personas(frame):
    global conteo_dentro, conteo_fuera

    if p1 == (-1, -1) or p2 == (-1, -1):
        return frame

    x_min, y_min = min(p1[0], p2[0]), min(p1[1], p2[1])
    x_max, y_max = max(p1[0], p2[0]), max(p1[1], p2[1])

    resultados = model(frame)[0]
    personas = [det for det in resultados.boxes if int(det.cls) == 0]

    conteo_dentro = 0
    conteo_fuera = 0

    for box in personas:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        centro_x = (x1 + x2) // 2
        centro_y = (y1 + y2) // 2

        if x_min <= centro_x <= x_max and y_min <= centro_y <= y_max:
            conteo_dentro += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            conteo_fuera += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
    cv2.putText(frame, "SALIDA", (p1[0], p1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Dentro: {conteo_dentro}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Fuera: {conteo_fuera}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

# ======================
# Guardar snapshot
# ======================
def guardar_snapshot(frame):
    if not os.path.exists("resultados_salidas"):
        os.makedirs("resultados_salidas")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resultados_salidas/snapshot_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Snapshot guardado: {filename}")
    guardar_registros_mongo(conteo_dentro, conteo_fuera, frame, timestamp)

# ======================
# Inicializar cámara
# ======================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

cv2.namedWindow("Monitor de Salidas")
cv2.setMouseCallback("Monitor de Salidas", dibujar_rectangulo)

print("Instrucciones:")
print("1. Seleccione el área de salida con el mouse")
print("2. Presione ESPACIO para snapshot manual")
print("3. El sistema guarda capturas cada 30s si hay detección")
print("4. Presione ESC para salir")

ultima_captura = time.time()

# ======================
# Bucle principal
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    imagen_copia = frame.copy()

    if area_seleccionada:
        frame = contar_personas(frame)
        tiempo_actual = time.time()
        if tiempo_actual - ultima_captura >= intervalo_captura:
            guardar_snapshot(frame)
            ultima_captura = tiempo_actual
    else:
        cv2.putText(frame, "Seleccione el area de salida", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if p1 != (-1, -1) and p2 != (-1, -1):
            cv2.rectangle(frame, p1, p2, (255, 255, 0), 1)

    cv2.imshow("Monitor de Salidas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # ESPACIO
        guardar_snapshot(frame)
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
