import cv2
# import numpy as np

print(cv2.useOptimized())
e1 = cv2.getTickCount()


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    e3 = cv2.getTickCount()
    while ret:
        c1 = cv2.getTickCount()
        ret, frame = cap.read()
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray", output)
        cv2.imshow("Live Feed", frame)
        print((cv2.getTickCount() - c1) * 1000 / cv2.getTickFrequency(), "ms")
        if cv2.waitKey(1) == ord('q'):
            break
    print(((e3 - e1) / cv2.getTickFrequency()), "s")
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
