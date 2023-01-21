import imutils
import cv2

IMAGE_FILE = 'IMG_6075.JPG'
img = cv2.imread(IMAGE_FILE)
new_height = 512
resized_img = imutils.resize(img, height=new_height)
cv2.imwrite('base_image.jpg', resized_img)