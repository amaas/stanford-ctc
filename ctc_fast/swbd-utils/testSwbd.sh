#!/bin/bash

# verbose
set -x

base=/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk
swbd=$base/egs/swbd/s5b
infile=/afs/cs.stanford.edu/u/awni/ctc_fast/$1
#data=dev
data=eval2000
numfiles=1

echo $infile
python ../runNNet.py --maxUttLen 1550 --dataDir  $swbd/exp/${data}_ctc/ --numFiles $numfiles --inFile $infile --test --outputDim 35 --inputDim $((41*15)) --rawDim $((41*15)) 
#python mergecharsswbd.py
#$base/src/bin/compute-wer --text --mode=present ark:$swbd/data/$data/text_ctc.filt ark:mergehypswbd.txt #>& wer.txt
