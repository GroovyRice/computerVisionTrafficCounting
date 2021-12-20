import cv2
import numpy as np

img = np.zeros((200, 200, 3), np.uint8)
height, width, channels = img.shape
img[:int(height / 2)] = [138, 25, 55]
img[int(height / 2):] = [0, 22, 0]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("AFTER GRAYSCALE")
print("Top Half = ", img[0][0])
print("Bottom Half = ", img[150][0])
ret, img = cv2.threshold(img, 12, 255, cv2.THRESH_TOZERO)
print("AFTER THRESHOLD")
print("Top Half = ", img[0][0])
print("Bottom Half = ", img[150][0])
cv2.imshow('', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Black is 0 in grayscale and White is 255
# cv2.threshold(IMAGE,MinValue,MaxValue,cv2.THRESH_BINARY)
#   When the grayscale is within the bounds of [MinValue,MaxValue]
#   then whatever that value is will go to the MaxValue. For example
#   if a pixel is 56 on grayscale and the bounds are [24,89] this
#   pixel will become 89 but if its 21 it will become zero.
#
# cv2.THRESH_BINARY_INV will inverse what has occurred, if it within
#   the bounds it will go to zero, outside the threshold it will go
#   to MaxValue
#
# cv2.THRESH_TRUNC outside the threshold will always stay the same but
#   rather than going to MaxValue if within the threshold you will go
#   to MinValue
#
# cv2.THRESH_TOZERO within the threshold will stay the same outside the
#   threshold will go to zero
# cv2.THRESH_TOZERO_INV the opposite of the THRESH_TOZERO; inside threshold
#   go to zero outside stay the same
