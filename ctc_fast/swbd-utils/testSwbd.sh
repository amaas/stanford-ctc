#!/bin/bash

# verbose
set -x

base=/scail/group/deeplearning/speech/zxie/kaldi-stanford/kaldi-trunk
swbd=$base/egs/swbd/s5b
infile=/afs/cs.stanford.edu/u/zxie/kaldi-stanford/stanford-nnet/ctc_fast/models/swbd_5_1824_bitemporal_3_step_1e-5_mom_.95_anneal_1.3.bin
#data=dev
data=eval2000
numfiles=23

echo $infile
#python ../runNNet.py --maxUttLen 1550 --dataDir  $swbd/exp/${data}_ctc/ --numFiles $numfiles --inFile $infile --test --outputDim 35 --inputDim $((41*15)) --rawDim $((41*15)) 
#python ../runDecode.py --dataDir $swbd/exp/${data}_ctc/  --numFiles $numfiles
python mergechars.py
python convert_to_ctm.py
./score_sclite.sh sclite_score hyp.ctm
grep Sum sclite_score/eval2000.ctm.filt.sys | /scail/group/deeplearning/speech/zxie/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/utils/best_wer.sh
