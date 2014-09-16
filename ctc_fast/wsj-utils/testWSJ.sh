#!/bin/bash

# Best WSJ with maxGNorm = 5000
# verbose
set -x

base=/scail/group/deeplearning/speech/zxie/kaldi-stanford/kaldi-trunk
wsj=$base/egs/wsj/s6
infile=/afs/cs.stanford.edu/u/zxie/kaldi-stanford/stanford-nnet/ctc_fast/models/wsj_5_1824_bitemporal_3_step_1e-5_mom_.95_anneal_1.2.bin
#data=train_si284
#data=test_dev93
data=test_eval92
numfiles=2

echo $infile
python ../runNNet.py --dataDir $wsj/exp/${data}_ctc/ --numFiles $numfiles --inFile $infile --test 
python mergechars.py
$base/src/bin/compute-wer --text --mode=present ark:$wsj/data/$data/text_ctc ark:mergehyp.txt #>& wer.txt
