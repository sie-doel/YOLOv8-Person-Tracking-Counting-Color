from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Tracker initialization
model.track(conf=0.4, persist=True)

# Dictionary for color memory per ID
id_to_color = {}

# Color extraction helper
def get_dominant_color(image, k=3):
    data = image.reshape((-1, 3))
    data = np.float32(data)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(data)
    colors = kmeans.cluster_centers_.astype(int)
    labels = np.bincount(kmeans.labels_)
    dominant = colors[np.argmax(labels)]
    return tuple(int(c) for c in dominant)

# Color to readable text
def color_to_name(rgb):
    r, g, b = rgb
    if r > 150 and g < 100 and b < 100:
        return "Merah"
    elif g > 150 and r < 100 and b < 100:
        return "Hijau"
    elif b > 150 and r < 100 and g < 100:
        return "Biru"
    elif r > 150 and g > 150 and b < 100:
        return "Kuning"
    elif r > 150 and g > 150 and b > 150:
        return "Putih"
    elif r < 80 and g < 80 and b < 80:
        return "Hitam"
    else:
        return f"RGB{rgb}"

# Webcam capture
cap = cv2.VideoCapture(0)

unique_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, persist=True, conf=0.4, show=False, verbose=False)[0]

    if results.boxes.id is not None:
        ids = results.boxes.id.cpu().numpy().astype(int)
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            person_crop = frame[y1:y2, x1:x2]

            if track_id not in id_to_color:
                if person_crop.size > 0:
                    dominant_rgb = get_dominant_color(person_crop)
                    color_name = color_to_name(dominant_rgb)
                    id_to_color[track_id] = color_name

            color_name = id_to_color.get(track_id, "Unknown")

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {track_id} | {color_name}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            unique_ids.add(track_id)

    # Show people count
    cv2.putText(frame, f"Orang terdeteksi: {len(unique_ids)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("YOLOv8 + Tracking + Warna Baju", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
