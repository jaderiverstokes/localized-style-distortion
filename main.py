import cv2
import numpy as np
import os


# possible styles to choose from
styles = [ 'candy', 'mosaic', 'starry-night', 'udnie']
style = styles[3]
# name of input inamge (no extension)
input_name = 'logan'

# Run image segmentation and stylization
command ='source mask.sh '+input_name+' && source style.sh '+input_name + ' ' + style
os.system(command)


# filenames
input_str = 'images/content-images/'+input_name+'.jpg'
style_str = 'images/style-transferred-images/'+input_name+'-'+style+'.jpg'
mask_str = 'images/mask-images/'+input_name+'-mask.png'
bin_mask_str = 'images/mask-images/'+input_name+'-bin-mask.png'
blurredMaskFN = input_name+'-blurred-mask.png'
blurMaskFP = 'images/mask-images/'+blurredMaskFN


# Create a binary mask from the image segmentation
input = cv2.imread(input_str)
styled = cv2.imread(style_str)
styled = styled.astype(float)
input = input.astype(float)

mask = cv2.imread(mask_str)
size = (720,1080,1)
m = np.zeros(size, dtype=np.uint8)
m[np.where((mask == [128,128,192]).all(axis = 2))] = 255

# Create a blurred alpha-mask from the binary mask
cv2.imwrite(bin_mask_str, m)
blurSigma = 50
command = './ImageTools/blurMask '+bin_mask_str +' '+str(blurSigma)+' ' + blurredMaskFN
os.system(command)
m = cv2.imread(blurMaskFP).astype(float)/255.0

# apply alpha blending
style_layer = cv2.multiply(m, styled)
regular_layer = cv2.multiply(1.0-m, input)
out = style_layer + regular_layer

#save output image
output_str = 'images/output-images/'+input_name+'-masked-'+style+'.png'
cv2.imwrite(output_str,out)
