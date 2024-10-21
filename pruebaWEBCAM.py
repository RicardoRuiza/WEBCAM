import random
import cv2
import numpy as np
from ultralytics import YOLO

# Abrir el archivo en modo lectura
my_file = open("utils/coco.txt", "r")
# Leer el archivo
data = my_file.read()
# Separar el texto cuando se encuentra un salto de línea ('\n')
class_list = data.split("\n")
my_file.close()

# Generar colores aleatorios para la lista de clases
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Cargar el modelo YOLOv8n preentrenado
model = YOLO("weights/yolov8n.pt", "v8")

# Valores para redimensionar los frames del video | un frame más pequeño optimiza la ejecución
frame_wid = 640
frame_hyt = 480

# Cambiar la fuente de captura a la webcam
cap = cv2.VideoCapture(1)  

if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    # Si no se lee el frame correctamente, 'ret' será False

    if not ret:
        print("No se puede recibir el frame (¿Fin de la transmisión?). Saliendo ...")
        break

    # Redimensionar el frame | optimizar la ejecución con un tamaño menor
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predecir en la imagen
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Convertir el array tensor a numpy
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # retorna una caja
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Mostrar nombre de clase y confianza
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Mostrar el frame resultante
    cv2.imshow("ObjectDetection", frame)

    # Terminar la ejecución al presionar la tecla "Q"
    if cv2.waitKey(1) == ord("q"):
        break

# Cuando todo esté hecho, liberar la captura
cap.release()
cv2.destroyAllWindows()
