#!/bin/bash

# verbose
set -x

momentum=.95 
epochs=20
layerSize=2048
numLayers=5
maxBatch=2000
step=1e-5
anneal=1.5
deviceId=0
temporalLayer=3

outfile="models/wsj_layers_${numLayers}_${layerSize}_temporal_${temporalLayer}_step_${step}_mom_${momentum}_anneal_${anneal}.bin"
echo $outfile
python runNNet.py --layerSize $layerSize --numLayers $numLayers \
  --step $step --epochs $epochs --momentum $momentum --anneal $anneal \
  --outputDim 34 --inputDim $((41*23)) --rawDim $((41*23))  --numFiles 75 \
  --dataDir /scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/wsj/s6/exp/train_si284_ctc/ \
  --maxUttLen $maxBatch --outFile $outfile --deviceId $deviceId --temporalLayer $temporalLayer
