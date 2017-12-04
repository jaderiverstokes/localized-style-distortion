export STYLE=$1
python fast-neural-style/neural_style/neural_style.py train --dataset ../train2014 --style-image images/style-images/$STYLE --vgg-model-dir fast-neural-style/neural_style/ --save-model-dir images/save-models/ --epochs 2 --cuda 1
