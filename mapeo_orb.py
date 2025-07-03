import cv2
import os
import numpy as np

# Crear carpeta para almacenar escenas si no existe
escena_dir = "escenas_guardadas"
os.makedirs(escena_dir, exist_ok=True)

# Inicializar ORB y BruteForce matcher
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Lista de escenas cargadas desde disco
stored_scenes = []

# Cargar escenas guardadas
for file in os.listdir(escena_dir):
    if file.endswith(".jpg"):
        nombre = file[:-4]  # sin extensión
        img_path = os.path.join(escena_dir, f"{nombre}.jpg")
        des_path = os.path.join(escena_dir, f"{nombre}_des.npy")

        if os.path.exists(des_path):
            img = cv2.imread(img_path)
            des = np.load(des_path, allow_pickle=True)
            stored_scenes.append((des, img, nombre))
            print(f"[✓] Escena cargada: {nombre}")

# Iniciar cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises y extraer keypoints
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    frame_kp = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)

    # Buscar coincidencias con escenas guardadas
    best_match_name = None
    best_match_count = 0

    for i, (stored_des, _, name) in enumerate(stored_scenes):
        if stored_des is None or des is None:
            continue

        matches = bf.match(des, stored_des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > best_match_count and len(matches) > 30:
            best_match_name = name
            best_match_count = len(matches)

    # Mostrar resultado
    if best_match_name:
        cv2.putText(frame_kp, f"Reconocido: {best_match_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Mapeo visual con ORB", frame_kp)

    key = cv2.waitKey(1) & 0xFF

    # Guardar nueva escena al presionar 's'
    if key == ord('s'):
        escena_id = len(os.listdir(escena_dir)) // 2 + 1
        nombre = f"escena_{escena_id}"
        img_path = os.path.join(escena_dir, f"{nombre}.jpg")
        des_path = os.path.join(escena_dir, f"{nombre}_des.npy")

        cv2.imwrite(img_path, frame)
        np.save(des_path, des)

        stored_scenes.append((des, frame.copy(), nombre))
        print(f"[✓] Escena guardada como {nombre}")

    # Salir con 'q'
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
