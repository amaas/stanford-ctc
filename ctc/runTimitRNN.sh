#!/bin/bash

# verbose
set -x

optimizer=nesterov
momentum=.95 
epochs=100
layers=2048,2048
step=1e-4
anneal=1.1
# temporal layer is 1-indexed <=0 means no temporal
temporal_layer=2


outfile="models/${optimizer}_layers_${layers}_step_${step}_mom_${momentum}__rec_${temporal_layer}_anneal.bin"
echo $outfile
python runNNet.py --layers $layers --optimizer $optimizer --step $step \
  --epochs $epochs --momentum $momentum --outFile $outfile --anneal $anneal \
  --temporal_layer $temporal_layer\
  --outputDim 62 --inputDim $((41*23)) --rawDim $((41*23))  --numFiles 19 \
  --dataDir /scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/timit/s5/exp/nn_train/

