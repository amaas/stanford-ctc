#!/bin/bash

# verbose
set -x

optimizer=nesterov
momentum=.95 
epochs=1
layers=1024,1024
step=1e-4

outfile=models/$optimizer_step_$step_mom_$momentum.bin

python runNNet.py --layers $layers --optimizer $optimizer --step $step \
  --epochs $epochs --momentum $momentum --outFile $outfile

