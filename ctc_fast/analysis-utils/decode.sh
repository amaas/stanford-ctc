#!/bin/bash

# verbose
set -x

base=/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk
swbd=$base/egs/wsj/s6
infile=/afs/cs.stanford.edu/u/awni/ctc/$1
outfile=/afs/cs.stanford.edu/u/awni/ctc/$2
maxBatch=1500
data=test_eval92
numfiles=2

mkdir $outfile

python writeLikelihoods.py --dataDir  $swbd/exp/${data}_ctc/ --numFiles $numfiles --inFile $infile \
                  --outFile $outfile --maxUttLen $maxBatch
                  
