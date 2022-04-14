import cv2
import numpy as np

footage = "footage_1.mp4"
print("Optimized:", cv2.useOptimized())
init_cap = cv2.VideoCapture(footage)

width = init_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = init_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Width:", width)
print("Height:", height)

mainFrame = np.zeros((int(height) * 2, int(width), 3), np.uint8)
temp = np.zeros((int(height), int(width), 3), np.uint8)
# Creates a background subtraction... using "detectShadows=True" within brackets
# Will allow for shadows to also be detected.
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Initialise Kernels
kernelOpen = np.ones((3, 3), np.uint8)
kernelClose = np.ones((8, 8), np.uint8)

#BIG DICK
dick= 5

# Region of Interest
Rx = np.multiply([0.1, 0.8], width).astype(int)
Ry = np.multiply([0.35, 1], height).astype(int)

detect = []
pos = int((Ry[1] - Ry[0]) * 0.6)
error = 6


def main():
    cap = cv2.VideoCapture(footage)
    counter = 0
    validator = 0
    while cap.isOpened():
        c1 = cv2.getTickCount()
        ret, frame = cap.read()
        ROI = frame[Ry[0]:Ry[1], Rx[0]:Rx[1]]
        if ROI is None:
            break
        fgMask = backSub.apply(ROI)
        if ret:
            # Binary Thresholding
            ret, imageBinary = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)

            # Erosion followed by dilation (Morphology)
            mask = cv2.morphologyEx(imageBinary, cv2.MORPH_OPEN, kernelOpen)

            # Dilation followed by Erosion (Morphology)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelClose)

            # Gaussian Blur
            mask = cv2.medianBlur(mask, 5)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for i in contours:
                area = cv2.contourArea(i)
                if area > (width * height / 150):
                    x1, y1, x2, y2 = cv2.boundingRect(i)
                    m = cv2.moments(i)
                    cX = int(m['m10'] / m['m00'])
                    cY = int(m['m01'] / m['m00'])
                    x1 += Rx[0]
                    y1 += Ry[0]
                    detect.append((cX, cY))
                    for (x, y) in detect:
                        if (pos - error) < y < (pos + error) and validator > 10:
                            counter += 1
                            print("Counter: ", counter)
                            cv2.line(frame, (Rx[0], Ry[0] + pos), (Rx[1], Ry[0] + pos), (255, 127, 0), 10)
                            detect.remove((x, y))
                            validator = 0
                        else:
                            validator += 1
                            detect.remove((x, y))
                    cv2.rectangle(frame, (x1, y1), (x1 + x2, y1 + y2), (0, 255, 0), 3)
                    cv2.circle(frame, (cX + Rx[0], cY + Ry[0]), 5, (255, 0, 0), -1)
            cv2.rectangle(frame, (Rx[0], Ry[0]), (Rx[1], Ry[1]), (0, 0, 255), 1)
            cv2.line(frame, (Rx[0], Ry[0] + pos), (Rx[1], Ry[0] + pos), (0, 0, 255), 2)
            cv2.putText(frame, "VEHICLE COUNT : " + str(counter), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 32, 255),
                        5)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            temp[Ry[0]:Ry[1], Rx[0]:Rx[1]] = mask
            finalFrame = np.concatenate((frame, temp), axis=0)
            finalFrame = cv2.resize(finalFrame, (0, 0), fx=0.6, fy=0.6)
            cv2.imshow("Live Feed", finalFrame)
            print((cv2.getTickCount() - c1) * 1000 / cv2.getTickFrequency(), "ms")
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
