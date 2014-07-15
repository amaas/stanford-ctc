#!/bin/bash

# verbose
set -x

momentum=.95 
epochs=20
layerSize=1824
numLayers=5
maxBatch=1500
step=1e-5
anneal=1.3
deviceId=0
temporalLayer=3

outfile="models/swbd_layers_${numLayers}_${layerSize}_bitemporal_${temporalLayer}_step_${step}_mom_${momentum}_anneal_${anneal}.bin"
echo $outfile
python runNNet.py --layerSize $layerSize --numLayers $numLayers \
  --step $step --epochs $epochs --momentum $momentum --anneal $anneal \
  --outputDim 35 --inputDim $((41*15)) --rawDim $((41*15))  --numFiles 384\
  --dataDir /scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/train_ctc/ \
  --maxUttLen $maxBatch --outFile $outfile --deviceId $deviceId --temporalLayer $temporalLayer
