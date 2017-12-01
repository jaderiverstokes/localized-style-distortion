import cv2
import numpy as np
input = cv2.imread('images/content-images/diego.jpg')
styled = cv2.imread('images/output-images/diego-starry-night.jpg')
mask = cv2.imread('images/mask-images/mask.png')
m = np.zeros((720,1080,1), dtype=np.uint8)
m[np.where((mask == [128,128,192]).all(axis = 2))] = 255
style_layer = cv2.bitwise_and(styled, styled, mask=m)
regular_layer = cv2.bitwise_and(input, input, mask=255-m)
out = style_layer + regular_layer
cv2.imwrite('logan.png',out)
