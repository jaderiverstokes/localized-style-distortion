export INPUT=$1
pyenv local 2.7.9
python tensorflow-deeplab-resnet/inference.py fast-neural-style/images/content-images/$INPUT.jpg models/deeplab_resnet.ckpt --save-dir ./images/mask-images/$INPUT.jpg
pyenv local 3.5.0
