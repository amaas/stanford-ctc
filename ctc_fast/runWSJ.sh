#!/bin/bash

# verbose
set -x
# wsj_layers_5_1824_bitemporal_3_step_1e-5_mom_.95_anneal_1.2
momentum=.95 
epochs=20
layerSize=3000
numLayers=5
maxBatch=2000
step=1e-5
anneal=1.2
deviceId=0
temporalLayer=3

outfile="models/wsj_layers_${numLayers}_${layerSize}_bitemporal_${temporalLayer}_step_${step}_mom_${momentum}_anneal_${anneal}.bin"
echo $outfile
python runNNet.py --layerSize $layerSize --numLayers $numLayers \
  --step $step --epochs $epochs --momentum $momentum --anneal $anneal \
  --outputDim 29 --inputDim $((41*23)) --rawDim $((41*23))  --numFiles 75 \
  --dataDirFeat /scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/wsj/s6/exp/train_si284_ctc/ \
  --dataDirAli /scail/group/deeplearning/speech/amaas/kaldi-stanford/kaldi-trunk/egs/wsj/s6/exp/train_si284_ctc/ \
  --maxUttLen $maxBatch --outFile $outfile --deviceId $deviceId --temporalLayer $temporalLayer
