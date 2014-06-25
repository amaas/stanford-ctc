#!/bin/bash

# verbose
set -x

base=/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk
wsj=$base/egs/wsj/s6
infile=/afs/cs.stanford.edu/u/awni/ctc/$1
#data=train_si284
data=test_dev93
numfiles=1

echo $infile
python runNNet.py --dataDir $wsj/exp/${data}_ctc/ --numFiles $numfiles --inFile $infile --test 

#$base/src/bin/compute-wer --text --mode=present ark:$timit/data/$data/text ark:hyp.txt >& wer.txt
