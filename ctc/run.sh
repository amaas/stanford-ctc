#!/bin/bash

# verbose
set -x

optimizer=nesterov
momentum=.95 
epochs=8
layers=1024,1024
step=1e-4
anneal=1.8

outfile="models/${optimizer}_step_${step}_mom_${momentum}_anneal.bin"
echo $outfile
python runNNet.py --layers $layers --optimizer $optimizer --step $step \
  --epochs $epochs --momentum $momentum --outFile $outfile --anneal $anneal

