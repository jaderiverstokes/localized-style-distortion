# inception time

import cv2
import numpy as np
import sys, os
from config import *

if(len(sys.argv)<4):
	print("Error: <input> <style> <iters>")
input_name = sys.argv[1]
s = sys.argv[2]

n = int(sys.argv[3])

zoom = 1.05

command = 'python fast-neural-style/neural_style/neural_style.py eval --content-image images/style-transferred-images/'+input_name+'/'+input_name+'-'+s+'.jpg --model images/saved-models/'+s+'.pth --output-image images/style-transferred-images/'+input_name+'/'+input_name+'-inception-'+s+'.jpg --cuda 0'
suffix = "-inception"
name_in =input_name + suffix
fin = 'images/style-transferred-images/'+input_name+'/'+input_name+'-'+s+'.jpg'
orig = cv2.imread(fin)
h, w,_ = orig.shape
ho = int(h*zoom)
wo = int(w*zoom)
fot = 'images/style-transferred-images/'+input_name+'/'+input_name+'-inception-'+s+'.jpg'
for i in range(n):
	print("Generating: " +input_name + "-" + s + ", " + str(i+1))
	os.system(command)
	inc = cv2.imread(fot)	
	inc = cv2.resize(inc, (ho, wo))
	inc = inc[0:h, 0:w]
	cv2.imwrite(fot, inc)
	command = 'python fast-neural-style/neural_style/neural_style.py eval --content-image '+fot+' --model images/saved-models/'+s+'.pth --output-image images/style-transferred-images/'+input_name+'/'+input_name+'-inception-'+s+'.jpg --cuda 0'
	