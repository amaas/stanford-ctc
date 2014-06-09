#!/bin/bash

# verbose
set -x

optimizer=nesterov
momentum=.95 
epochs=100
layers=2048,2048,2048,2048
step=1e-4
anneal=1.1

outfile="models/${optimizer}_layers_${layers}_step_${step}_mom_${momentum}_anneal.bin"
echo $outfile
python runNNet.py --layers $layers --optimizer $optimizer --step $step \
  --epochs $epochs --momentum $momentum --outFile $outfile --anneal $anneal \
  --outputDim 62 --inputDim $((41*23)) --rawDim $((41*23))  --numFiles 19 \
  --dataDir /scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/timit/s5/exp/nn_train/

