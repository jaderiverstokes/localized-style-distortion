# Generates all permutations of 2 styles (for foreground / background)

import cv2
import numpy as np
import sys, os

if(len(sys.argv)<2):
	print("Error: Please specify an input image")

input_name = sys.argv[1]

styles = [ 
	'candy', 
	'mosaic', 
	'starry-night', 
	'udnie'	]

out_dir = "images/output-images/" + input_name
command = "mkdir -p " + out_dir
os.system(command)

style_file = lambda s: 'images/style-transferred-images/' +input_name+ '/' +input_name+ '-' +s+ '.jpg'
mask_file = 'images/mask-images/'+input_name+'-mask.png'
bin_mask_file = 'images/mask-images/'+input_name+'-bin-mask.png'
blur_mask_file = 'images/mask-images/'+input_name+'-blurred-mask.png'

out_file = lambda s1, s2: 'images/output-images/'+input_name+'/'+input_name+'-masked-'+ s1 +'-'+ s2 +'.png'


blur_sigma = 10
blur_size  = 2*blur_sigma+1


mask = cv2.imread(mask_file)
h,w = mask.shape[:2]
size = (h,w,1)
bin_mask = np.zeros(size, dtype=np.uint8)
bin_mask[np.where((mask == [128,128,192]).all(axis = 2))] = 255
cv2.imwrite(bin_mask_file, bin_mask)
blur_mask = cv2.GaussianBlur(bin_mask, (blur_size, 	blur_size), blur_sigma)
cv2.imwrite(blur_mask_file, blur_mask)

m = cv2.imread(blur_mask_file).astype(float) / 255.0


for s1 in styles:
	im1 = cv2.imread(style_file(s1)).astype(float)
	fg = cv2.multiply(m, im1)
	for s2 in styles:
		if s1 is s2:
			continue
		im2 = cv2.imread(style_file(s2)).astype(float)
		bg = cv2.multiply(1.0-m, im2)
		im_out = fg + bg
		cv2.imwrite(out_file(s1, s2), im_out)

