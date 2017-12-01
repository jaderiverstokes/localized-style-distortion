export INPUT=$1
pyenv local 2.7.9
python tensorflow-deeplab-resnet/inference.py images/content-images/$INPUT.jpg ../mask-models/deeplab_resnet.ckpt --save-dir ./images/mask-images/$INPUT-
pyenv local 3.5.0
