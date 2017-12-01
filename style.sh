export INPUT=diego
export STYLE=starry-night
python ../fast-neural-style/neural_style/neural_style.py eval --content-image images/content-images/$INPUT.jpg --model ../fast-neural-style/saved-models/$STYLE.pth --output-image images/output-images/$INPUT-$STYLE.jpg --cuda 0
