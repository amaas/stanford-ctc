#!/bin/bash

kaldi_dir=/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b

. $kaldi_dir/path.sh ## Source the tools/utils 

for x in 'dev' 'eval2000'
do
    dir=exp/${x}_ctc #location to store nn data and models
    data=data/$x #location of data
    train_dir=exp/train_ctc #location of data using to train 
    echo "writing" $data

    transform=$train_dir/$(readlink $train_dir/final.feature_transform)
    echo $transform

    mkdir $dir 2>/dev/null

    feats="ark:copy-feats scp:$data/feats.scp ark:- | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"

    feat-write --utts-per-file=200 --feature-transform=$transform "$feats" $kaldi_dir/$dir/
done


