"""
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Define detection areas (adjust coordinates as needed)
area1 = np.array([(256, 142), (566, 139), (572, 248), (265, 261)], np.int32)
area2 = np.array([(274, 359), (575, 356), (576, 476), (271, 477)], np.int32)

# Load YOLOv8 model (choose appropriate model and path)
model = YOLO("yolov8m.pt")  # Replace with your model path

def mouse_callback(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_MOUSEMOVE:
        img_with_text = frame.copy()
        cv2.putText(img_with_text, f"X: {x}, Y: {y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("FRAME", img_with_text)

def main():
    global frame
    frame = cv2.imread('input/mahdi_parking.jpg')  # Replace with your image path
    frame = cv2.resize(frame, (1280, 720))
    cv2.polylines(frame, [area1, area2], True, (0, 255, 0), 3)

    results = model(frame)

    # Check results format and access bounding boxes
    if hasattr(results, 'xyxy'):
        boxes = results.xyxy[0]  # Assuming first element contains detections
    else:
        print("Results format not recognized. Check YOLOv8 version and documentation.")
        return  # Exit if bounding boxes cannot be accessed

    car_in_area1 = False
    car_in_area2 = False

    for box in boxes:
        xmin, ymin, xmax, ymax, conf, cls = box.tolist()  # Extract box coordinates
        if model.names[int(cls)] == 'car' and conf > 0.5:  # Adjust confidence threshold
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)
            inside_area1 = cv2.pointPolygonTest(area1, (cx, cy), False) >= 0
            inside_area2 = cv2.pointPolygonTest(area2, (cx, cy), False) >= 0
            if inside_area1:
                car_in_area1 = True
            if inside_area2:
                car_in_area2 = True

    if car_in_area1:
        cv2.polylines(frame, [area1], True, (0, 0, 255), 3)
    if car_in_area2:
        cv2.polylines(frame, [area2], True, (0, 0, 255), 3)

    cv2.namedWindow("FRAME")
    cv2.setMouseCallback("FRAME", mouse_callback)
    cv2.imshow("FRAME", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

"""


"""
#------------------------------------Cas Video Importe--------------------------------------------------------
import cv2
import torch
import numpy as np

# Pour stocker les coordonner des points de curseur cliqués
points = []

# A chaque fois que le curseur de la souris est déplacé, on imprime les coordonnées du nouveau point
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Importation de la vidéo à utiliser
cap = cv2.VideoCapture('input/parking.mp4')
count=0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 600))

    results = model(frame)
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = (row['name'])
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        if 'car' in d:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, str(d), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("FRAME", frame)
    cv2.setMouseCallback("FRAME", POINTS)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()


#------------------------------------------------------------------------------------------------------------
"""

# -------------------------------------------Cas de Web Cam Phone-------------------------------------------
"""
import cv2
import torch
import numpy as np

points = []

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

def authenticate(username, password):
    # Vérifie si le nom d'utilisateur et le mot de passe correspondent aux informations fournies
    return username == 'anass' and password == 'anass'

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Nom d'utilisateur et mot de passe
username = 'anass'
password = 'anass'

# Vérification des informations d'identification
if not authenticate(username, password):
    print("Accès refusé. Vérifiez vos informations d'identification.")
    exit()

# Accès à la caméra du téléphone via son adresse IP
url = 'http://anass:anass@192.168.11.109:8080/video'
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la caméra.")
    exit()


area=[(),(),(),()]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))

    results = model(frame)
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = (row['name'])
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        if 'car' in d:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, str(d), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("FRAME", frame)
    cv2.setMouseCallback("FRAME", POINTS)

    if cv2.waitKey(1) == 27:  # Attendre 1ms pour la touche ESC (27)
        break

cap.release()
cv2.destroyAllWindows()
"""
# ------------------------------------------------------------------------------------------------------------


# ---------------------------------------------Cas Images-----------------------------------------------------
"""
import cv2
import torch
import numpy as np

# Fonction callback pour les événements de la souris
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Afficher les coordonnées du curseur sur la console
        print(f"Coordonnées du curseur : {x}, {y}")

# Définition des zones
#area1 = np.array([(602, 309), (683, 309), (691, 352), (604, 348)], np.int32)
#area2 = np.array([(678, 291), (753, 291), (787, 355), (696, 354)], np.int32)

area1 = np.array([(470, 157), (568, 153), (568, 194), (469, 194)], np.int32)
area2 = np.array([(467, 278), (571, 278), (572, 317), (469, 320)], np.int32)


# Chargement du modèle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Chargement et préparation de l'image
image_path = 'input/Capture d’écran 2024-03-23 143259.jpg'  # Remplacez par le chemin de votre image
frame = cv2.imread(image_path)
frame = cv2.resize(frame, (1020, 600))

# Initialisation de la fenêtre d'affichage et du callback de la souris
cv2.namedWindow('Result')
cv2.setMouseCallback('Result', POINTS)

# Dessin des contours initiaux des zones en vert
cv2.polylines(frame, [area1, area2], True, (0, 255, 0), 3)

# Détection des objets dans l'image
results = model(frame)

# Indicateurs de détection dans les zones
car_in_area1 = False
car_in_area2 = False

# Traitement des résultats de la détection
for index, row in results.pandas().xyxy[0].iterrows():
    x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
    d, cx, cy = row['name'], (x1 + x2) // 2, (y1 + y2) // 2

    if 'car' in d:
        if cv2.pointPolygonTest(area1, (cx, cy), False) >= 0: car_in_area1 = True
        if cv2.pointPolygonTest(area2, (cx, cy), False) >= 0: car_in_area2 = True

# Mise à jour des contours en fonction des détections
if car_in_area1: cv2.polylines(frame, [area1], True, (0, 0, 255), 3)
if car_in_area2: cv2.polylines(frame, [area2], True, (0, 0, 255), 3)

# Affichage de l'image
cv2.imshow('Result', frame)
cv2.waitKey(0)  # Attente d'une touche pour fermer
cv2.destroyAllWindows()



#------------------------------------------------------------------------------------------------------------
"""

from flask import Flask
import cv2
import torch
import numpy as np
import requests
import time
from datetime import datetime

app = Flask(__name__)

# Définir les zones de détection dans une liste pour un accès plus facile
areas = [
    np.array([(451, 456), (590, 410), (781, 566), (643, 621)], np.int32),
    np.array([(605, 408), (725, 367), (915, 512), (784, 563)], np.int32),
    np.array([(738, 364), (842, 335), (1024, 471), (925, 510)], np.int32),
    np.array([(905, 140), (1040, 112), (1164, 253), (1025, 281)], np.int32)
]

# Charger un modèle YOLOv5 pour la détection d'objets
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

def mouse_callback(event, x, y, flags, param):
    # Afficher les coordonnées du curseur dans la console lors du mouvement de la souris
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Coordonnées du curseur: x={x}, y={y}")

def main():
    # Ouvrir la capture vidéo
    cap = cv2.VideoCapture('input/par.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps * 7)

    # Nom de la fenêtre pour les rappels de la souris
    cv2.namedWindow("FRAME")
    cv2.setMouseCallback("FRAME", mouse_callback)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            frame = cv2.resize(frame, (1280, 720))
            for area in areas:
                cv2.polylines(frame, [area], True, (0, 255, 0), 3)

            results = model(frame)
            cars_in_areas = [False] * len(areas)

            for _, row in results.pandas().xyxy[0].iterrows():
                if 'car' in row['name']:
                    cx = int((row['xmin'] + row['xmax']) / 2)
                    cy = int((row['ymin'] + row['ymax']) / 2)
                    for i, area in enumerate(areas):
                        if cv2.pointPolygonTest(area, (cx, cy), False) >= 0:
                            cars_in_areas[i] = True

            for i, car_in_area in enumerate(cars_in_areas):
                if car_in_area:
                    cv2.polylines(frame, [areas[i]], True, (0, 0, 255), 3)

            cv2.imshow("FRAME", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            json_data = {'timestamp': timestamp, 'positions': {f'a{i+1}': car for i, car in enumerate(cars_in_areas)}}
            try:
                requests.post("http://192.168.1.106:5001/json_receive", json=json_data, timeout=10)
            except requests.exceptions.RequestException as e:
                print(f"Error sending data: {e}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


