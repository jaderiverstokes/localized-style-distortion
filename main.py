import cv2
import numpy as np
import os

styles = [ 'candy', 'mosaic', 'starry-night', 'udnie']
input_name = 'logan'
style = styles[3]
command ='source mask.sh '+input_name+' && source style.sh '+input_name + ' ' + style
# os.system(command)
input_str = 'images/content-images/'+input_name+'.jpg'
style_str = 'images/style-transferred-images/'+input_name+'-'+style+'.jpg'
mask_str = 'images/mask-images/'+input_name+'-mask.png'
input = cv2.imread(input_str)
styled = cv2.imread(style_str)
mask = cv2.imread(mask_str)
m = np.zeros((720,1080,1), dtype=np.uint8)
m[np.where((mask == [128,128,192]).all(axis = 2))] = 255
style_layer = cv2.bitwise_and(styled, styled, mask=m)
regular_layer = cv2.bitwise_and(input, input, mask=255-m)
out = style_layer + regular_layer
output_str = 'images/output-images/'+input_name+'-masked-'+style+'.png'
cv2.imwrite(output_str,out)
