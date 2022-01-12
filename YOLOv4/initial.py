import cv2
import numpy as np


# INITIAL VARIABLE DECLARATION
footage_name = "P1060692.MP4"
scale = 0.00392
conf_threshold = 0.4
nms_threshold = 0.3


# OPENING DNN WITH OPENCV
with open('coco.names', 'r') as file:
    classes = [cname.strip() for cname in file.readlines()]
COLOURS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')


# FUNCTION FOR GETTING OUTPUT LAYERS
def get_output_layers():
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


# DRAWING OF BOUNDING BOXES
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    colour = COLOURS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), colour, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)


# INITIALISE PARAMETERS OF FOOTAGE
def init_footage(footage):
    print("Optimized:", cv2.useOptimized())
    init_cap = cv2.VideoCapture(footage)
    width = init_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = init_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Width:", width)
    print("Height:", height)
    return width, height


# DETECTION AND MAPPING
def run_inference(image):
    width = image.shape[1]
    height = image.shape[0]
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers())
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], int(x), int(y), int(x + w), int(y + h))


# MAIN FUNCTION
def main():
    w, h = init_footage(footage_name)
    cap = cv2.VideoCapture(footage_name)
    count = 0
    while True:
        print("Choose a Frame Rate:")
        temp = input()
        if temp.isnumeric():
            mod = int(temp)
            break
        print("Incorrect input {" + temp + "}")
    while True:
        print("Would you like to do it in ROI (Y/N):")
        temp = input()
        if temp.upper() == 'Y':
            Rx = np.multiply([0.1, 0.8], w).astype(int)
            Ry = np.multiply([0.35, 1], h).astype(int)
            break
        elif temp.upper() == 'N':
            Rx = np.multiply([0, 1], w).astype(int)
            Ry = np.multiply([0, 1], h).astype(int)
            break
        print("Incorrect input {" + temp + "}")
    while cap.isOpened():
        ret, frame = cap.read()
        ROI = frame[Ry[0]:Ry[1], Rx[0]:Rx[1]]
        if ROI is None:
            break
        count += 1
        if count % mod != 0:
            continue
        c1 = cv2.getTickCount()
        run_inference(ROI)
        c2 = cv2.getTickCount()
        cv2.imshow("Live Feed", frame)
        print((cv2.getTickCount() - c1) * 1000 / cv2.getTickFrequency(), "ms")
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
