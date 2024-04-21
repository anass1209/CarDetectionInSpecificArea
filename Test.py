"""
#-Avec ligne de commande execution
import cv2
import torch
import numpy as np

# Définir les zones de détection dans une liste pour un accès plus facile
areas = [
    np.array([(99, 329), (289, 326), (295, 421), (99, 435)], np.int32),
    np.array([(472, 355), (603, 350), (607, 413), (473, 418)], np.int32)
]

# Charger un modèle YOLOv5 pour la détection d'objets
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def main():
    # Demander à l'utilisateur s'il souhaite effectuer la détection
    response = input("Voulez-vous effectuer la détection de voitures ? (O/N) ")

    # Ouvrir la capture vidéo
    # Utilisation d'une caméra IP
    camera_url = 'http://anass:anass@192.168.1.102:8080/video'
    cap = cv2.VideoCapture(camera_url)

    # Changer la résolution de la caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Nouvelle largeur
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Nouvelle hauteur

    # Nom de la fenêtre pour les rappels de la souris
    cv2.namedWindow("FRAME")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Afficher les zones de détection sur le frame
        for area in areas:
            cv2.polylines(frame, [area], True, (0, 255, 0), 3)

        # Vérification lorsque 'Y:' est entré dans la console
        if response.upper() == 'O':
            # Réinitialiser l'état des places de parking à chaque itération
            parking_status = [False] * len(areas)

            # Effectuer la détection d'objets uniquement lorsque 'Y:' est entré dans la console
            results = model(frame)

            for i, area in enumerate(areas):
                # Vérifier si une voiture est détectée dans la zone de parking
                for _, row in results.pandas().xyxy[0].iterrows():
                    if 'car' in row['name']:
                        cx = int((row['xmin'] + row['xmax']) / 2)
                        cy = int((row['ymin'] + row['ymax']) / 2)
                        if cv2.pointPolygonTest(area, (cx, cy), False) >= 0:
                            parking_status[i] = True
                            cv2.polylines(frame, [area], True, (0, 0, 255), 3)  # Colorer la zone de parking en rouge si occupée
                            break

            for i, status in enumerate(parking_status):
                cv2.putText(frame, f"Place {i + 1}: {'occupée' if status else 'vide'}", (10, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if status else (0, 255, 0), 2)

            # Afficher les résultats dans la console
            print("État des emplacements :")
            for i, status in enumerate(parking_status):
                print(f"Place {i + 1}: {'occupée' if status else 'vide'}")

            # Reposer la question de savoir si l'utilisateur veut voir l'état des emplacements
            response = input("Voulez-vous effectuer la détection de voitures ? (O/N) ")

        cv2.imshow("FRAME", frame)

        # Vérification lorsque 'Q:' est entré dans la console
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
"""






#-------------------------------------Le Code Distribution Direct Avec Flux --------------------------------------------
import cv2
import torch
import numpy as np
import time
import requests
from datetime import datetime

# Définir les zones de détection dans une liste pour un accès plus facile
areas = [
    np.array([(99, 329), (289, 326), (295, 421), (96, 435)], np.int32),
    np.array([(472, 355), (603, 350), (607, 413), (473, 418)], np.int32)
]

# Charger un modèle YOLOv5 pour la détection d'objets
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def send_parking_status(ip_address, parking_status):
    url = f"http://{ip_address}:5000/update_status"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {'parking_status': parking_status, 'timestamp': current_time}
    requests.post(url, json=data)

def main():
    global parking_status
    # Ouvrir la capture vidéo
    # Utilisation d'une caméra IP
    camera_url = 'http://anass:anass@192.168.1.102:8080/video'
    cap = cv2.VideoCapture(camera_url)

    # Changer la résolution de la caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Nouvelle largeur
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Nouvelle hauteur

    # Nom de la fenêtre pour les rappels de la souris
    cv2.namedWindow("FRAME")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for area in areas:
            cv2.polylines(frame, [area], True, (0, 255, 0), 3)

        results = model(frame)

        # Réinitialiser l'état des places de parking à chaque itération
        parking_status = [False] * len(areas)

        for i, area in enumerate(areas):
            # Vérifier si une voiture est détectée dans la zone de parking
            for _, row in results.pandas().xyxy[0].iterrows():
                if 'car' in row['name']:
                    cx = int((row['xmin'] + row['xmax']) / 2)
                    cy = int((row['ymin'] + row['ymax']) / 2)
                    if cv2.pointPolygonTest(area, (cx, cy), False) >= 0:
                        parking_status[i] = True
                        cv2.polylines(frame, [area], True, (0, 0, 255), 3)  # Colorer la zone de parking en rouge si occupée
                        break

        for i, status in enumerate(parking_status):
            cv2.putText(frame, f"Place {i + 1}: {'occupée' if status else 'vide'}", (10, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if status else (0, 255, 0), 2)

        cv2.imshow("FRAME", frame)

        # Envoi des données à l'autre PC
        send_parking_status('192.168.1.106', parking_status)  # Remplacez '192.168.1.103' par l'adresse IP de l'autre PC

        # Vérification lorsque 'R:' est entré dans la console
        if cv2.waitKey(1) & 0xFF == 27:  # Appuyez sur la touche 'Escape' pour quitter
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()





"""
#--------------------------------Pour la partie de Affichage des cordonneees dans le console---------------------------
import cv2
import torch
import numpy as np
import time
import requests

# Définir les zones de détection dans une liste pour un accès plus facile
areas = [
    np.array([(99, 329), (289, 326), (295, 421), (96, 435)], np.int32),
    np.array([(472, 355), (603, 350), (607, 413), (473, 418)], np.int32)
]

# Charger un modèle YOLOv5 pour la détection d'objets
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Coordonnées de la souris : x={x}, y={y}")

def main():
    global parking_status
    # Ouvrir la capture vidéo
    # Utilisation d'une caméra IP
    camera_url = 'http://anass:anass@192.168.1.102:8080/video'
    cap = cv2.VideoCapture(camera_url)

    # Changer la résolution de la caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Nouvelle largeur
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Nouvelle hauteur

    # Nom de la fenêtre pour les rappels de la souris
    cv2.namedWindow("FRAME")
    cv2.setMouseCallback("FRAME", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for area in areas:
            cv2.polylines(frame, [area], True, (0, 255, 0), 3)

        results = model(frame)

        # Réinitialiser l'état des places de parking à chaque itération
        parking_status = [False] * len(areas)

        for i, area in enumerate(areas):
            # Vérifier si une voiture est détectée dans la zone de parking
            for _, row in results.pandas().xyxy[0].iterrows():
                if 'car' in row['name']:
                    cx = int((row['xmin'] + row['xmax']) / 2)
                    cy = int((row['ymin'] + row['ymax']) / 2)
                    if cv2.pointPolygonTest(area, (cx, cy), False) >= 0:
                        parking_status[i] = True
                        cv2.polylines(frame, [area], True, (0, 0, 255), 3)  # Colorer la zone de parking en rouge si occupée
                        break

        for i, status in enumerate(parking_status):
            cv2.putText(frame, f"Place {i + 1}: {'occupée' if status else 'vide'}", (10, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if status else (0, 255, 0), 2)

        cv2.imshow("FRAME", frame)

        # Vérification lorsque 'R:' est entré dans la console
        if cv2.waitKey(1) & 0xFF == 27:  # Appuyez sur la touche 'Escape' pour quitter
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------------------------------------------------
"""