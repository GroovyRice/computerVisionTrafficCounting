import cv2
import numpy as np
import torch
from pathlib import Path
from polygon import *
import statistics
import math
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
objects = [0, 2, 3, 5, 7]  # person, car, motorbike, bus and truck
# CHECK coco.names to add more objects for detection.
confidence = 0.6
counter = []
counted = []
time = []
main_count = [0, 0, 0]
timer = 0
ROI = np.array([[400, 900], [1700, 900], [1420, 500], [720, 500]], np.int32)
LANE_1 = np.array([[400, 900], [850, 900], [970, 500], [720, 500]], np.int32)
LANE_2 = np.array([[970, 500], [1180, 500], [1260, 900], [850, 900]], np.int32)
LANE_3 = np.array([[1180, 500], [1420, 500], [1700, 900], [1260, 900]], np.int32)

with open('coco.names', 'r') as file:
    classes = [cname.strip() for cname in file.readlines()]
COLOURS = np.random.uniform(0, 255, size=(len(classes), 3))


def index_2d(new_list, v):
    for j, x in enumerate(new_list):
        if v in x:
            if x.index(v) == 0:
                return j


while cap.isOpened():
    c1 = cv2.getTickCount()
    ret, frame = cap.read()
    results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.polylines(frame, [LANE_1], True, color=(127, 127, 0), thickness=3)
    cv2.polylines(frame, [LANE_2], True, color=(127, 127, 0), thickness=3)
    cv2.polylines(frame, [LANE_3], True, color=(127, 127, 0), thickness=3)
    detections = results.pred[0].cpu().numpy()
    vehicles = []
    for i in detections:
        if i[5] in objects and i[4] > confidence:
            vehicles.append(i)
    vehicles = np.array(vehicles)
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
        if within_polygon([cent[1][0], cent[1][1]], ROI) and cent[0] not in counted:
            idx = index_2d(counter, cent[0])
            if idx is not None:
                counter[idx][1] += 1
                if counter[idx][1] > 70:
                    if within_polygon([cent[1][0], cent[1][1]], LANE_1):
                        main_count[0] += 1
                        cv2.polylines(frame, [LANE_1], True, color=(127, 255, 0), thickness=3)
                    if within_polygon([cent[1][0], cent[1][1]], LANE_2):
                        main_count[1] += 1
                        cv2.polylines(frame, [LANE_2], True, color=(127, 255, 0), thickness=3)
                    if within_polygon([cent[1][0], cent[1][1]], LANE_3):
                        main_count[2] += 1
                        cv2.polylines(frame, [LANE_3], True, color=(127, 255, 0), thickness=3)
                    counted.append(cent[0])
                    print(f"[COUNT] {classes[cent[2]]} of ID: {int(cent[0])}")
            else:
                counter.append([cent[0], 0])
    cv2.putText(frame, f"TOTAL COUNT: {main_count[0]+main_count[1]+main_count[2]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=3)
    cv2.putText(frame, f"LANE (1): {main_count[0]}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=3)
    cv2.putText(frame, f"LANE(2): {main_count[1]}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=3)
    cv2.putText(frame, f"LANE (3): {main_count[2]}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=3)

    time.append((cv2.getTickCount() - c1) * 1000 / cv2.getTickFrequency())
    if timer > 1000:
        timer = 0
        av = statistics.mean(time)
        realtime = av*60
        if realtime < 1000:
            RT = f"Real-Time Operation achieved; ahead by {realtime:.2f}"
        else:
            RT = f"Not Real-Time; behind by {realtime-1000:.2f}ms"
        print(f"Average Latency of Program: {av:.2f}ms\n{RT}")
    else:
        timer += 1
    cv2.imshow("LIVE", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
print(main_count)
