import cv2
import numpy as np
import os


# possible styles to choose from
styles = [ 'candy', 'mosaic', 'starry-night', 'udnie']
style = styles[0]
# name of input inamge (no extension)
input_name = 'hotpot'

# Run image segmentation and stylization
command ='source mask.sh '+input_name+' && source style.sh '+input_name + ' ' + style
#os.system(command)


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
h,w = input.shape[:2]
size = (h,w,1)
m = np.zeros(size, dtype=np.uint8)
m[np.where((mask == [128,128,192]).all(axis = 2))] = 255

# Create a blurred alpha-mask from the binary mask
cv2.imwrite(bin_mask_str, m)
blurSigma = 30
#command = './ImageTools/blurMask '+bin_mask_str +' '+str(blurSigma)+' ' + blurredMaskFN
#os.system(command)
#m = cv2.imread(blurMaskFP).astype(float)/255.0
m = cv2.imread(bin_mask_str).astype(float)/255.0
m = cv2.GaussianBlur(m, (2*blurSigma+1, 2*blurSigma+1), blurSigma)
cv2.imwrite(blurMaskFP, m)

# apply alpha blending
style_layer = cv2.multiply(m, styled)
regular_layer = cv2.multiply(1.0-m, input)
out = style_layer + regular_layer

#save output image
output_str = 'images/output-images/'+input_name+'-masked-'+style+'.png'
cv2.imwrite(output_str,out)



## Testing
extra = False

if(extra):
	#blend stylized foreground with different stylized background
	name = input_name
	style1 = styles[0]
	style2 = styles[3]
	style3 = styles[2]

	mask_fn = 'images/mask-images/'+name+'-bin-mask.png'
	mask = cv2.imread(mask_fn).astype(float)
	m = cv2.GaussianBlur(mask, (45, 45), 13)
	m = m/255.0

	im1_fn = 'images/style-transferred-images/'+name+'-'+style1+'.jpg'
	im1 = cv2.imread(im1_fn).astype(float)
	fg = cv2.multiply(m, im1)

	im2_fn = 'images/style-transferred-images/'+name+'-'+style2+'.jpg'
	im2 = cv2.imread(im2_fn).astype(float)
	bg = cv2.multiply(1.0-m, im2)

	im3_fn = 'images/style-transferred-images/'+name+'-'+style3+'.jpg'
	im3 = cv2.imread(im3_fn).astype(float)
	bg2 = cv2.multiply(1.0-m, im3)

	im_out = fg + bg*0.8 + bg2*0.2
	out_fn = 'images/output-images/'+name+'-masked-'+style1+'-'+style2+'-'+style3+'.png'
	cv2.imwrite(out_fn, im_out)

