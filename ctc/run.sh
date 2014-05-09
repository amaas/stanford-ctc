#!/bin/bash

# verbose
set -x

optimizer=nesterov
momentum=.95 
epochs=5
layers=1024,1024
step=1e-4
anneal=2

outfile="models/${optimizer}_step_${step}_mom_${momentum}.bin"
echo $outfile
python runNNet.py --layers $layers --optimizer $optimizer --step $step \
  --epochs $epochs --momentum $momentum --outFile $outfile --anneal $anneal

