import cv2
import torch
from pathlib import Path

from Main_Files.sort.sort import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
FOOTAGE = str(ROOT) + "/YOLOv4/P1060692.MP4"

cap = cv2.VideoCapture(FOOTAGE)

ret, frame = cap.read()
height, width, _ = frame.shape

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

mot_tracker = Sort()
counter = []
counted = []
main_count = 0

with open('coco.names', 'r') as file:
    classes = [cname.strip() for cname in file.readlines()]
COLOURS = np.random.uniform(0, 255, size=(len(classes), 3))


def index_2d(new_list, v):
    for j, x in enumerate(new_list):
        if v in x:
            if x.index(v) == 0:
                return j


while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    cv2.rectangle(frame, (1700, 900), (500, 500), color=(127, 200, 0), thickness=1)
    detections = results.pred[0].cpu().numpy()
    track_bbs_ids = mot_tracker.update(detections)
    centroids = []
    for i, TBI in enumerate(track_bbs_ids.tolist()):
        ID = TBI[8]
        label_idx = int(TBI[4])
        label = classes[label_idx]
        colour = COLOURS[label_idx]
        x1, y1, x2, y2 = int(TBI[0]), int(TBI[1]), int(TBI[2]), int(TBI[3])
        centre = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
        centroids.append([ID, centre, label_idx])
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness=2)
        cv2.putText(frame, f"{label}: {int(ID)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour,
                    thickness=2)
        cv2.circle(frame, tuple(centre), radius=1, color=(0, 0, 255), thickness=4)
    for i, cent in enumerate(centroids):
        if int(cent[2]) not in [2, 3, 5, 7]:
            continue
        if 1700 > cent[1][0] > 500 and 900 > cent[1][1] > 500 and cent[0] not in counted:
            idx = index_2d(counter, cent[0])
            if idx is not None:
                counter[idx][1] += 1
                if counter[idx][1] > 70:
                    main_count += 1
                    counted.append(cent[0])
                    print(f"[COUNT] {classes[cent[2]]} of ID: {int(cent[0])}")
            else:
                counter.append([cent[0], 0])

    cv2.putText(frame, f"COUNT: {main_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=3)
    cv2.imshow("LIVE", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
