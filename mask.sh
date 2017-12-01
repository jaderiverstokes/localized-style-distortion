export INPUT=$1
python2 tensorflow-deeplab-resnet/inference.py images/content-images/$INPUT.jpg ../mask-models/deeplab_resnet.ckpt --save-dir ./images/mask-images/$INPUT-

