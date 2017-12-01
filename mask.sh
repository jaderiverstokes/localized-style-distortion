export INPUT=diego.jpg
python inference.py ../fast-neural-style/images/content-images/$INPUT deeplab_resnet.ckpt --save-dir ./images/mask-images/$INPUT
