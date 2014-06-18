#!/bin/bash

# verbose
set -x

momentum=.95 
epochs=50
layerSize=2048
numLayers=4
maxBatch=777
step=1e-4
anneal=1.1
deviceId=0
temporalLayer=-1

outfile="models/timit_layers_${numLayers}_${layerSize}_step_${step}_mom_${momentum}_anneal_${anneal}.bin"

echo $outfile
python runNNet.py --layerSize $layerSize --numLayers $numLayers --step $step \
  --epochs $epochs --momentum $momentum --outFile $outfile --anneal $anneal \
  --outputDim 62 --inputDim $((41*23)) --rawDim $((41*23))  --numFiles 19 \
  --dataDir /scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/timit/s5/exp/nn_train/ --deviceId $deviceId --temporalLayer $temporalLayer 

