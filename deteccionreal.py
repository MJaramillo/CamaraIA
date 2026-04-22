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




cliente = pymongo.MongoClient("mongodb://localhost:27017/")
db = cliente["AFORO"]


# Variables globales para el ID del usuario activo y Aforo Máximo

GLOBAL_ACTIVE_USER_ID = None
GLOBAL_AFORO_MAXIMO = 10 # Valor por defecto
CHECK_ACTIVE_USER_INTERVAL = 5 # segundos
CHECK_AFORO_MAXIMO_INTERVAL = 10 # segundos


model = YOLO("yolov8n.pt")
model.conf = 0.5    # umbral de confianza


# Variables globales para selección de área

p1 = (-1, -1)
p2 = (-1, -1)
area_seleccionada = False
imagen_copia = None

# Variables de conteo
conteo_total = 0
conteo_dentro_emergencia = 0

# Tiempo de la última captura guardada
ultima_captura_automatica = 0
intervalo_captura = 30    # segundos

# Tiempo de la última vez que se verificó el usuario activo y aforo máximo
ultimo_check_usuario_activo = 0
ultimo_check_aforo_maximo = 0


# Convertir imagen a base64 para guardar y enviar

def convertir_a_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# Función para manejar eventos del mouse para selección de área

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
                print("Zona de emergencia configurada")


# Detectar y contar personas (total y dentro del área seleccionada)

def contar_personas(frame):
    global conteo_total, conteo_dentro_emergencia

    if p1 == (-1, -1) or p2 == (-1, -1):
        return frame, 0, 0

    x_min, y_min = min(p1[0], p2[0]), min(p1[1], p2[1])
    x_max, y_max = max(p1[0], p2[0]), max(p1[1], p2[1])

    resultados = model(frame, verbose=False)[0]
    personas = [det for det in resultados.boxes if int(det.cls) == 0] 

    conteo_total = len(personas)
    conteo_dentro_emergencia = 0

    for box in personas:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        centro_x = (x1 + x2) // 2
        centro_y = (y1 + y2) // 2

        if x_min <= centro_x <= x_max and y_min <= centro_y <= y_max:
            conteo_dentro_emergencia += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Verde para personas dentro de la zona
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Rojo para personas fuera de la zona

    # Dibujar rectángulo del área
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
    cv2.putText(frame, "SALIDA", (p1[0], p1[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Mostrar conteos
    cv2.putText(frame, f"Total Personas: {conteo_total}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # Amarillo para el total
    cv2.putText(frame, f"Zona de Emergencia: {conteo_dentro_emergencia}", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Verde para dentro de emergencia

    return frame, conteo_total, conteo_dentro_emergencia


# Función para verificar el usuario activo en la base de datos

def verificar_usuario_activo():
    global GLOBAL_ACTIVE_USER_ID, ultimo_check_usuario_activo
    
    tiempo_actual = time.time()
    if tiempo_actual - ultimo_check_usuario_activo >= CHECK_ACTIVE_USER_INTERVAL:
        try:
            active_session = db.sesiones_yolo_activas.find_one({"activo": True})
            
            if active_session and "user_id" in active_session:
                new_user_id = str(active_session["user_id"])
                if new_user_id != GLOBAL_ACTIVE_USER_ID:
                    GLOBAL_ACTIVE_USER_ID = new_user_id
                    print(f"\nScript de YOLO ahora monitoreando para el usuario: {GLOBAL_ACTIVE_USER_ID}")
            else:
                if GLOBAL_ACTIVE_USER_ID is not None:
                    print("\nNingún usuario activo encontrado en la web. El script NO enviará datos.")
                GLOBAL_ACTIVE_USER_ID = None
        except Exception as e:
            print(f"Error al verificar usuario activo en MongoDB: {e}")
            GLOBAL_ACTIVE_USER_ID = None
        finally:
            ultimo_check_usuario_activo = tiempo_actual

#obtener el aforo máximo desde la base de datos
def obtener_aforo_maximo_desde_db():
    global GLOBAL_AFORO_MAXIMO, ultimo_check_aforo_maximo, GLOBAL_ACTIVE_USER_ID

    tiempo_actual = time.time()
    if GLOBAL_ACTIVE_USER_ID and (tiempo_actual - ultimo_check_aforo_maximo >= CHECK_AFORO_MAXIMO_INTERVAL):
        try:
            config_data = db.CONFIGURACION.find_one(
                {"user_id": ObjectId(GLOBAL_ACTIVE_USER_ID), "nombre": "aforo_maximo"}
            )
            if config_data and "valor" in config_data:
                try:
                    new_aforo_maximo = int(config_data["valor"])
                    if new_aforo_maximo != GLOBAL_AFORO_MAXIMO:
                        GLOBAL_AFORO_MAXIMO = new_aforo_maximo
                        print(f"Aforo máximo actualizado a: {GLOBAL_AFORO_MAXIMO}")
                except ValueError:
                    print(f"Advertencia: El valor de 'aforo_maximo' en la DB no es un número válido: {config_data['valor']}")
            else:
                if GLOBAL_AFORO_MAXIMO != 10: # Si no se encuentra configuración, restablecer al valor por defecto
                    print("No se encontró aforo máximo para el usuario activo, usando valor por defecto: 10")
                    GLOBAL_AFORO_MAXIMO = 10
        except ObjectId.InvalidId: 
            print(f"Error: GLOBAL_ACTIVE_USER_ID '{GLOBAL_ACTIVE_USER_ID}' no es un ObjectId válido.")
        except Exception as e:
            print(f"Error al obtener aforo máximo de MongoDB: {e}")
        finally:
            ultimo_check_aforo_maximo = tiempo_actual
    elif not GLOBAL_ACTIVE_USER_ID:
        if GLOBAL_AFORO_MAXIMO != 10:
            GLOBAL_AFORO_MAXIMO = 10
        ultimo_check_aforo_maximo = tiempo_actual


# Función para enviar datos a la API de Flask (registros de aforo y capturas)

def enviar_a_flask_api(total, dentro_emergencia, imagen_b64, current_timestamp):
    global GLOBAL_ACTIVE_USER_ID

    if GLOBAL_ACTIVE_USER_ID is None:
        return 

    api_url = "http://127.0.0.1:5000/api/update_data" 
    payload = {
        "user_id": GLOBAL_ACTIVE_USER_ID,
        "total_personas": total,
        "personas_en_zona_emergencia": dentro_emergencia,
        "imagen_base64": imagen_b64,
        "timestamp": current_timestamp.strftime("%Y-%m-%d %H:%M:%S") 
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status() # Esto lanzará una excepción para errores 4xx/5xx

    except requests.exceptions.ConnectionError as e:
        print(f"ERROR: No se pudo conectar a la API de Flask. Asegúrate de que app.py esté corriendo en {api_url}. Error: {e}")
    except requests.exceptions.HTTPError as e:
        print(f"ERROR HTTP al enviar datos a Flask API: {e}")
        if e.response is not None:
            print(f"Detalles de la respuesta del servidor: {e.response.text}")
    except Exception as e:
        print(f"ERROR inesperado al enviar datos a Flask API: {e}")

# Función para guardar datos localmente (capturas)
def guardar_datos_localmente(frame, total, dentro_emergencia, user_id=None, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now() # Fallback si no se provee, aunque idealmente siempre se pasará
    
    fecha_str = timestamp.strftime("%Y-%m-%d")
    hora_str = timestamp.strftime("%H%M%S")

    # Crear una carpeta para el usuario dentro de 'capturas'
    # y luego una carpeta para la fecha dentro de la del usuario
    base_capturas_folder = "static/capturas" 
    user_capturas_folder = os.path.join(base_capturas_folder, user_id if user_id else "sin_usuario")
    daily_capturas_folder = os.path.join(user_capturas_folder, fecha_str)
    
    os.makedirs(daily_capturas_folder, exist_ok=True)

    nombre_archivo_captura = os.path.join(daily_capturas_folder, f"snapshot_{hora_str}.jpg")
    cv2.imwrite(nombre_archivo_captura, frame)

    print(f"Snapshot local guardado en: {nombre_archivo_captura}")


# Inicializar cámara y ventana
cap = cv2.VideoCapture(0) # 0 para la webcam predeterminada
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara. Asegúrate de que esté conectada y no esté en uso.")
    sys.exit(1)

cv2.namedWindow("Monitor de Salidas")
cv2.setMouseCallback("Monitor de Salidas", dibujar_rectangulo)

print("Instrucciones:")
print("1. Seleccione la zona de emergencia con el ratón (clic y arrastrar).")
print("2. Presione ESPACIO para guardar un snapshot manual.")
print("3. Presione ESC para salir.")

ultima_captura_automatica = time.time()
ultimo_check_usuario_activo = time.time()
ultimo_check_aforo_maximo = time.time() # Inicializa el tiempo de chequeo de aforo máximo

# Bucle principal

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error leyendo el frame de la cámara. Terminando...")
        break

    frame = cv2.resize(frame, (800, 600))
    imagen_copia = frame.copy()

    total_personas = 0
    dentro_emergencia = 0

    # Primero, verifica si hay un usuario activo
    verificar_usuario_activo()
    # Luego, obtiene el aforo máximo para ese usuario 
    obtener_aforo_maximo_desde_db()

    if area_seleccionada:
        frame, total_personas, dentro_emergencia = contar_personas(frame)


        tiempo_actual_loop = time.time() 
        
        # Lógica para envío automático de datos y capturas
        if tiempo_actual_loop - ultima_captura_automatica >= intervalo_captura:
            current_timestamp = datetime.now() # <-- Obtener el timestamp aquí
            imagen_base64_para_envio = convertir_a_base64(frame)
            
            if GLOBAL_ACTIVE_USER_ID:
                enviar_a_flask_api(total_personas, dentro_emergencia, imagen_base64_para_envio, current_timestamp) 
                guardar_datos_localmente(frame, total_personas, dentro_emergencia, GLOBAL_ACTIVE_USER_ID, current_timestamp)
                print(f"Snapshot automático guardado y datos enviados para {GLOBAL_ACTIVE_USER_ID} - Total Personas: {total_personas}, Dentro Emergencia: {dentro_emergencia}")
            else:
                guardar_datos_localmente(frame, total_personas, dentro_emergencia, "sin_usuario_activo", current_timestamp) 
                print(f"Snapshot automático guardado localmente (sin usuario activo en la web) - Total Personas: {total_personas}, Dentro Emergencia: {dentro_emergencia}")
            
            ultima_captura_automatica = tiempo_actual_loop
    else:
        cv2.putText(frame, "Seleccione el area de salida (arrastre el raton)", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Monitor de Salidas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 32 and area_seleccionada:   
        frame, total_personas, dentro_emergencia = contar_personas(frame)
        current_timestamp = datetime.now() 
        imagen_base64_para_envio = convertir_a_base64(frame)
        
        if GLOBAL_ACTIVE_USER_ID:
            enviar_a_flask_api(total_personas, dentro_emergencia, imagen_base64_para_envio, current_timestamp) 
            guardar_datos_localmente(frame, total_personas, dentro_emergencia, GLOBAL_ACTIVE_USER_ID, current_timestamp) 
            print(f"Snapshot manual guardado y datos enviados para {GLOBAL_ACTIVE_USER_ID} - Total Personas: {total_personas}, Dentro Emergencia: {dentro_emergencia}")
            
        else:
            guardar_datos_localmente(frame, total_personas, dentro_emergencia, "sin_usuario_activo", current_timestamp) 
            print(f"Snapshot manual guardado localmente (sin usuario activo en la web) - Total Personas: {total_personas}, Dentro Emergencia: {dentro_emergencia}")

    elif key == 27:    # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()