from ultralytics import YOLO
import cv2

# Carga el modelo YOLO base (cuando tengas el tuyo, cámbialo por "frutas_yolo.pt")
model = YOLO("yolov8n.pt")

# Inicia la cámara (0 = cámara principal)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza la detección
    results = model(frame, verbose=False)

    # Dibuja los resultados directamente
    annotated_frame = results[0].plot()

    # Muestra el video
    cv2.imshow("Detección de frutas", annotated_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
