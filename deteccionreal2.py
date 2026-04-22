import os
from ultralytics import YOLO
import cv2
from datetime import datetime
import time
import base64
import pymongo
import requests
import sys
from bson.objectid import ObjectId

# Conexión a MongoDB
try:
    cliente = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    db = cliente["AFORO"]
    cliente.server_info()
    print("Conexión a MongoDB exitosa.")
except pymongo.errors.ServerSelectionTimeoutError as err:
    print(f"Error de conexión a MongoDB: {err}")
    sys.exit(1)

# Variables globales
GLOBAL_ACTIVE_USER_ID = None
GLOBAL_AFORO_MAXIMO = 10
CHECK_ACTIVE_USER_INTERVAL = 5
CHECK_AFORO_MAXIMO_INTERVAL = 10
INTERVALO_PROCESAMIENTO = 30  # <--- Nuevo: Intervalo en segundos para cambiar de imagen

# Carga del modelo YOLO
model = YOLO("yolov8n.pt")
model.conf = 0.5

# Variables para selección de área
p1, p2 = (-1, -1), (-1, -1)
area_seleccionada = False
imagen_copia = None

# Variables de tiempo
ultimo_check_usuario_activo = 0
ultimo_check_aforo_maximo = 0
ultimo_procesamiento_automatico = 0

# --- CONFIGURACIÓN PARA IMÁGENES DESDE CARPETA ---
IMAGE_FOLDER = r"C:\Users\Marcelo Jaramillo\Downloads\DeteccionPersonasCamara\2025-06-26"
if not os.path.exists(IMAGE_FOLDER):
    print(f"Error: La carpeta de imágenes '{IMAGE_FOLDER}' no existe.")
    sys.exit(1)

# --- FUNCIONES (sin cambios) ---
def convertir_a_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

def dibujar_rectangulo(event, x, y, flags, param):
    global p1, p2, area_seleccionada, imagen_copia
    if not area_seleccionada:
        if event == cv2.EVENT_LBUTTONDOWN:
            p1 = (x, y)
            p2 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and p1 != (-1, -1):
            p2 = (x, y)
            temp_img = imagen_copia.copy()
            cv2.rectangle(temp_img, p1, p2, (0, 0, 255), 2)
            cv2.imshow("Monitor de Salidas", temp_img)
        elif event == cv2.EVENT_LBUTTONUP and p1 != (-1, -1):
            p2 = (x, y)
            if p1 != p2:
                area_seleccionada = True
                print("✅ Zona de emergencia configurada. Iniciando procesamiento automático...")

def contar_personas(frame):
    if not area_seleccionada: return frame, 0, 0
    x_min, y_min = min(p1[0], p2[0]), min(p1[1], p2[1])
    x_max, y_max = max(p1[0], p2[0]), max(p1[1], p2[1])
    resultados = model(frame, verbose=False)[0]
    personas = [det for det in resultados.boxes if int(det.cls) == 0]
    conteo_total = len(personas)
    conteo_dentro_emergencia = 0
    for box in personas:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        centro_x, centro_y = (x1 + x2) // 2, (y1 + y2) // 2
        color = (0, 0, 255) if x_min <= centro_x <= x_max and y_min <= centro_y <= y_max else (0, 255, 0)
        if color == (0, 0, 255): conteo_dentro_emergencia += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv2.putText(frame, "SALIDA", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Total Personas: {conteo_total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Zona de Emergencia: {conteo_dentro_emergencia}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame, conteo_total, conteo_dentro_emergencia

def verificar_usuario_activo():
    global GLOBAL_ACTIVE_USER_ID, ultimo_check_usuario_activo
    if time.time() - ultimo_check_usuario_activo < CHECK_ACTIVE_USER_INTERVAL: return
    try:
        active_session = db.sesiones_yolo_activas.find_one({"activo": True})
        new_user_id = str(active_session["user_id"]) if active_session else None
        if new_user_id != GLOBAL_ACTIVE_USER_ID:
            GLOBAL_ACTIVE_USER_ID = new_user_id
            print(f"👤 Usuario activo: {GLOBAL_ACTIVE_USER_ID}" if GLOBAL_ACTIVE_USER_ID else "👤 Ningún usuario activo.")
    except Exception as e:
        print(f"Error al verificar usuario: {e}")
        GLOBAL_ACTIVE_USER_ID = None
    finally:
        ultimo_check_usuario_activo = time.time()

def obtener_aforo_maximo_desde_db():
    global GLOBAL_AFORO_MAXIMO, ultimo_check_aforo_maximo
    if not GLOBAL_ACTIVE_USER_ID or time.time() - ultimo_check_aforo_maximo < CHECK_AFORO_MAXIMO_INTERVAL: return
    try:
        config_data = db.CONFIGURACION.find_one({"user_id": ObjectId(GLOBAL_ACTIVE_USER_ID), "nombre": "aforo_maximo"})
        if config_data and "valor" in config_data:
            new_aforo = int(config_data["valor"])
            if new_aforo != GLOBAL_AFORO_MAXIMO:
                GLOBAL_AFORO_MAXIMO = new_aforo
                print(f"📊 Aforo máximo actualizado a: {GLOBAL_AFORO_MAXIMO}")
        else:
            GLOBAL_AFORO_MAXIMO = 10
    except Exception as e:
        print(f"Error al obtener aforo: {e}")
    finally:
        ultimo_check_aforo_maximo = time.time()

def enviar_a_flask_api(total, dentro, img_b64, ts):
    if not GLOBAL_ACTIVE_USER_ID: return
    try:
        payload = {"user_id": GLOBAL_ACTIVE_USER_ID, "total_personas": total, "personas_en_zona_emergencia": dentro, "imagen_base64": img_b64, "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")}
        requests.post("http://127.0.0.1:5000/api/update_data", json=payload, timeout=10).raise_for_status()
        print(f"🚀 Datos enviados a la API para el usuario {GLOBAL_ACTIVE_USER_ID}.")
    except requests.exceptions.RequestException as e:
        print(f"Error al enviar a la API: {e}")

def guardar_datos_localmente(frame, user_id, ts):
    user_folder = user_id if user_id else "sin_usuario"
    daily_folder = os.path.join("static", "capturas", user_folder, ts.strftime("%Y-%m-%d"))
    os.makedirs(daily_folder, exist_ok=True)
    filepath = os.path.join(daily_folder, f"snapshot_{ts.strftime('%H%M%S')}.jpg")
    cv2.imwrite(filepath, frame)
    print(f"💾 Snapshot guardado en: {filepath}")

# --- BUCLE PRINCIPAL AUTOMÁTICO ---
cv2.namedWindow("Monitor de Salidas")
cv2.setMouseCallback("Monitor de Salidas", dibujar_rectangulo)
image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if not image_files:
    print(f"No se encontraron imágenes en '{IMAGE_FOLDER}'. Saliendo.")
    sys.exit(0)

print("--- INSTRUCCIONES ---")
print("1. Seleccione la zona de emergencia con el ratón en la ventana.")
print("2. Una vez seleccionada, el script procesará una imagen cada 30 segundos.")
print("3. Presione 'ESC' para salir del programa.")

image_index = 0
frame_mostrado = None

while True:
    # La llamada a waitKey es esencial para que la ventana responda y detecte la tecla ESC
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print("Saliendo del programa...")
        break

    # Fase 1: Esperar a que el usuario seleccione el área
    if not area_seleccionada:
        image_path = os.path.join(IMAGE_FOLDER, image_files[0])
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error al leer la primera imagen '{image_path}'.")
            break
        frame = cv2.resize(frame, (1024, 768))
        imagen_copia = frame.copy() # Copia para la función de dibujo
        cv2.putText(frame, "POR FAVOR, SELECCIONE EL AREA DE SALIDA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Monitor de Salidas", frame)
        continue # Vuelve al inicio del bucle hasta que el área sea seleccionada

    # Fase 2: Ciclo de procesamiento automático
    tiempo_actual = time.time()
    if tiempo_actual - ultimo_procesamiento_automatico >= INTERVALO_PROCESAMIENTO:
        if image_index >= len(image_files):
            print("Todas las imágenes han sido procesadas. Saliendo.")
            break

        # Cargar y procesar la imagen actual
        image_path = os.path.join(IMAGE_FOLDER, image_files[image_index])
        print(f"\n--- Procesando imagen {image_index + 1}/{len(image_files)}: {image_files[image_index]} ---")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error al leer la imagen '{image_path}'. Saltando a la siguiente.")
            image_index += 1
            ultimo_procesamiento_automatico = tiempo_actual # Resetea el timer para no bloquearse
            continue
        
        frame = cv2.resize(frame, (1024, 768))
        
        # Verificar usuario y aforo
        verificar_usuario_activo()
        obtener_aforo_maximo_desde_db()

        # Realizar conteo
        frame_procesado, total, dentro = contar_personas(frame.copy())
        
        # Guardar y enviar datos
        current_timestamp = datetime.now()
        guardar_datos_localmente(frame_procesado, GLOBAL_ACTIVE_USER_ID, current_timestamp)
        if GLOBAL_ACTIVE_USER_ID:
            imagen_b64 = convertir_a_base64(frame_procesado)
            enviar_a_flask_api(total, dentro, imagen_b64, current_timestamp)

        frame_mostrado = frame_procesado # Actualiza el frame que se mostrará en la ventana
        image_index += 1
        ultimo_procesamiento_automatico = tiempo_actual

    # Muestra continuamente el último frame procesado durante la espera
    if frame_mostrado is not None:
        # Añadir un contador de tiempo en pantalla
        tiempo_restante = max(0, INTERVALO_PROCESAMIENTO - (time.time() - ultimo_procesamiento_automatico))
        texto_timer = f"Siguiente en: {tiempo_restante:.0f}"
        cv2.putText(frame_mostrado, texto_timer, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Monitor de Salidas", frame_mostrado)

cv2.destroyAllWindows()
cliente.close()
print("Recursos liberados. Programa terminado.")