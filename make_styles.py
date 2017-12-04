# This script will apply each style to create a bank of 
# stylized images so they can be computed all at once
# and also create the image segmentation mask

import sys
import os

if(len(sys.argv)<2):
	print("Error: Please specify an input image")

input_name = sys.argv[1]

out_dir = "images/style-transferred-images/" + input_name
command = "mkdir -p " + out_dir
os.system(command)

styles = [ 
	'candy', 
	'mosaic', 
	'starry-night', 
	'udnie'	]


for s in styles:
	command = 'source style2.sh '+input_name + ' ' + s
	print("Genereating: " + input_name + "-" + s)
	os.system(command)

print("Generating: segmentation")
command ='source mask.sh '+input_name
os.system(command)


