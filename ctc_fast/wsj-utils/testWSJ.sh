#!/bin/bash

# Best WSJ with maxGNorm = 5000
# verbose
set -x

base=/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk
wsj=$base/egs/wsj/s6
infile=/afs/cs.stanford.edu/u/awni/ctc_fast/$1
#data=train_si284
#data=test_dev93
data=test_eval92
numfiles=2

echo $infile
python ../runNNet.py --dataDir $wsj/exp/${data}_ctc/ --numFiles $numfiles --inFile $infile --test 
python mergechars.py
$base/src/bin/compute-wer --text --mode=present ark:$wsj/data/$data/text_ctc ark:mergehyp.txt #>& wer.txt
