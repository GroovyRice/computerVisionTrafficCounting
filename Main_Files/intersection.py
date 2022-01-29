import cv2
import numpy as np
import torch
from pathlib import Path
from polygon import *
import statistics
import math

from Main_Files.sort.sort import *


FOOTAGE = "P1060700.MP4"

cap = cv2.VideoCapture(FOOTAGE)

ret, frame = cap.read()
height, width, _ = frame.shape

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.eval()

mot_tracker = Sort()
objects = [0, 2, 3, 5, 7]  # person, car, motorbike, bus and truck
# CHECK coco.names to add more objects for detection.
confidence = 0.6
counter = []
counted = []
time = []
main_count = [[0, 0], [0, 0], [0, 0]]
timer = 0
INTER_1_LANE_OUT = np.array([[980, 780], [810, 1080], [500, 1080], [770, 760]], np.int32)
INTER_1_LANE_IN = np.array([[810, 1080], [980, 780], [1200, 820], [1100, 1080]], np.int32)
INTER_2_LANE_OUT = np.array([[785, 470], [275, 420], [200, 520], [710, 570]], np.int32)
INTER_2_LANE_IN = np.array([[700, 670], [200, 620], [200, 520], [710, 570]], np.int32)
INTER_3_LANE_OUT = np.array([[1140, 490], [1310, 510], [1390, 290], [1270, 270]], np.int32)
INTER_3_LANE_IN = np.array([[1140, 490], [980, 480], [1060, 270], [1200, 270]], np.int32)

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
    cv2.polylines(frame, [INTER_1_LANE_OUT], True, color=(127, 127, 0), thickness=3)
    cv2.polylines(frame, [INTER_1_LANE_IN], True, color=(127, 127, 0), thickness=3)
    cv2.polylines(frame, [INTER_2_LANE_OUT], True, color=(127, 127, 0), thickness=3)
    cv2.polylines(frame, [INTER_2_LANE_IN], True, color=(127, 127, 0), thickness=3)
    cv2.polylines(frame, [INTER_3_LANE_OUT], True, color=(127, 127, 0), thickness=3)
    cv2.polylines(frame, [INTER_3_LANE_IN], True, color=(127, 127, 0), thickness=3)
    detections = results.pred[0].cpu().numpy()
    vehicles = []
    for i in detections:
        if i[5] in objects and i[4] > confidence:
            vehicles.append(i)
    vehicles = np.array(vehicles)
    track_bbs_ids = mot_tracker.update(vehicles)
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
        if cent[0] in counted:
            continue
        if within_polygon([cent[1][0], cent[1][1]], INTER_1_LANE_IN) or within_polygon([cent[1][0], cent[1][1]], INTER_1_LANE_OUT):
            idx = index_2d(counter, cent[0])
            if idx is not None:
                counter[idx][1] += 1
                if counter[idx][1] > 40:
                    if within_polygon([cent[1][0], cent[1][1]], INTER_1_LANE_IN):
                        main_count[0][0] += 1
                        cv2.polylines(frame, [INTER_1_LANE_IN], True, color=(127, 255, 0), thickness=3)
                    elif within_polygon([cent[1][0], cent[1][1]], INTER_1_LANE_OUT):
                        main_count[0][1] += 1
                        cv2.polylines(frame, [INTER_1_LANE_OUT], True, color=(127, 255, 0), thickness=3)
                    counted.append(cent[0])
                    print(f"[COUNT] {classes[cent[2]]} of ID: {int(cent[0])}")
            else:
                counter.append([cent[0], 0])
        elif within_polygon([cent[1][0], cent[1][1]], INTER_2_LANE_IN) or within_polygon([cent[1][0], cent[1][1]], INTER_2_LANE_OUT):
            idx = index_2d(counter, cent[0])
            if idx is not None:
                counter[idx][1] += 1
                if counter[idx][1] > 50:
                    if within_polygon([cent[1][0], cent[1][1]], INTER_2_LANE_IN):
                        main_count[1][0] += 1
                        cv2.polylines(frame, [INTER_2_LANE_IN], True, color=(127, 255, 0), thickness=3)
                    elif within_polygon([cent[1][0], cent[1][1]], INTER_2_LANE_OUT):
                        main_count[1][1] += 1
                        cv2.polylines(frame, [INTER_2_LANE_OUT], True, color=(127, 255, 0), thickness=3)
                    counted.append(cent[0])
                    print(f"[COUNT] {classes[cent[2]]} of ID: {int(cent[0])}")
            else:
                counter.append([cent[0], 0])
        elif within_polygon([cent[1][0], cent[1][1]], INTER_3_LANE_IN) or within_polygon([cent[1][0], cent[1][1]], INTER_3_LANE_OUT):
            idx = index_2d(counter, cent[0])
            if idx is not None:
                counter[idx][1] += 1
                if counter[idx][1] > 50:
                    if within_polygon([cent[1][0], cent[1][1]], INTER_3_LANE_IN):
                        main_count[2][0] += 1
                        cv2.polylines(frame, [INTER_3_LANE_IN], True, color=(127, 255, 0), thickness=3)
                    elif within_polygon([cent[1][0], cent[1][1]], INTER_3_LANE_OUT):
                        main_count[2][1] += 1
                        cv2.polylines(frame, [INTER_3_LANE_OUT], True, color=(127, 255, 0), thickness=3)
                    counted.append(cent[0])
                    print(f"[COUNT] {classes[cent[2]]} of ID: {int(cent[0])}")
            else:
                counter.append([cent[0], 0])
    cv2.putText(frame, f"TOTAL COUNT: [IN {main_count[0][0]+main_count[1][0]+main_count[2][0]}, OUT {main_count[0][1]+main_count[1][1]+main_count[2][1]}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=3)
    cv2.putText(frame, f"INTER (1): [IN {main_count[0][0]}, OUT {main_count[0][1]}]",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=3)
    cv2.putText(frame, f"INTER (2): [IN {main_count[1][0]}, OUT {main_count[1][1]}]",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255),
                thickness=3)
    cv2.putText(frame, f"INTER (3): [IN {main_count[2][0]}, OUT {main_count[2][1]}]",
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
