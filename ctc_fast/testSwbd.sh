#!/bin/bash

# verbose
set -x

base=/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk
swbd=$base/egs/swbd/s5b
infile=/afs/cs.stanford.edu/u/awni/ctc/$1
data=train
numfiles=1

echo $infile
python runNNet.py --dataDir  $swbd/exp/${data}_ctc/ --numFiles $numfiles --inFile $infile --test \
                  --outputDim 35 --inputDim $((41*15)) --rawDim $((41*15)) 

#$base/src/bin/compute-wer --text --mode=present ark:$timit/data/$data/text ark:hyp.txt >& wer.txt
