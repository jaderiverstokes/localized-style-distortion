export INPUT=$1
export STYLE=$2
python fast-neural-style/neural_style/neural_style.py eval --content-image images/style-transferred-images/$INPUT/$INPUT.jpg --model images/saved-models/$STYLE.pth --output-image images/style-transferred-images/$INPUT/$INPUT-inception-$STYLE.jpg --cuda 0
