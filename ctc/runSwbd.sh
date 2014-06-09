#!/bin/bash

# verbose
set -x

optimizer=nesterov
momentum=.95 
epochs=6
layers=3000,3000,3000,3000,3000
step=5e-5
anneal=1.5
#infile=/afs/cs.stanford.edu/u/awni/ctc/models/nesterov_layers_2048,2048,2048,2048,2048_step_1e-5_mom_.95_anneal.bin

outfile="models/${optimizer}_layers_${layers}_step_${step}_mom_${momentum}_anneal.bin"
echo $outfile
python runNNet.py --layers $layers --optimizer $optimizer --step $step \
  --epochs $epochs --momentum $momentum --outFile $outfile --anneal $anneal \
  --outputDim 35 --inputDim $((41*15)) --rawDim $((41*15))  --numFiles 384 \
  --dataDir /scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/train_ctc/ \
  #--inFile $infile
