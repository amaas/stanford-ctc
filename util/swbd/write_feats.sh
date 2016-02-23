#!/bin/bash

# feature config
norm_vars=false # normalize the FBANKs (CVN)
splice_lr=20    # temporal splicing
splice_step=1   # stepsize of the splicing (1 is no gap between frames, just like splice_feats does)

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 

data=data/train_nodup
dir=exp/train_ctc
kaldi_dir=/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b

echo "$0 [info]: Build Training Data"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data \n"

mkdir -p $dir/{log,nnet}

# shuffle the list
echo "Preparing train lists"
cat $data/feats.scp | $kaldi_dir/utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp

# print the list sizes
wc -l $dir/train.scp

#get feature dim
echo -n "Getting feature dim : "
feat_dim=$(feat-to-dim --print-args=false scp:$dir/train.scp -)
echo $feat_dim

#read the features
feats="ark:copy-feats scp:$dir/train.scp ark:- |"

#add per-speaker CMVN
echo "Will use CMVN statistics : $data/cmvn.scp"
[ ! -r $data/cmvn.scp ] && echo "Cannot find cmvn stats $data/cmvn.scp" && exit 1;
cmvn="scp:$data/cmvn.scp"
feats="$feats apply-cmvn --print-args=false --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk $cmvn ark:- ark:- |"
# keep track of norm_vars option
echo "$norm_vars" >$dir/norm_vars 

# Generate the splice transform
echo "Using splice +/- $splice_lr , step $splice_step"
feature_transform=$dir/tr_splice$splice_lr-$splice_step.nnet
$kaldi_dir/utils/nnet/gen_splice.py --fea-dim=$feat_dim --splice=$splice_lr --splice-step=$splice_step > $feature_transform

# keep track of feat_type
echo $feat_type > $dir/feat_type

#renormalize the input to zero mean and unit variance
cmvn_g="$dir/cmvn_glob.mat"
echo "Renormalizing input features by : $cmvn_g"
compute-cmvn-stats --binary=false "$feats nnet-forward $feature_transform ark:- ark:- |" $cmvn_g 2>${cmvn_g}_log || exit 1
#convert the global cmvn stats to nnet format
cmvn-to-nnet --binary=false $cmvn_g $cmvn_g.nnet 2>$cmvn_g.nnet_log || exit 1;
#append matrix to feature_transform
{
feature_transform_old=$feature_transform
feature_transform=${feature_transform%.nnet}_cmvn-g.nnet
cp $feature_transform_old $feature_transform
cat $cmvn_g.nnet >> $feature_transform
}

###### MAKE LINK TO THE FINAL feature_transform, so the other scripts will find it ######
(cd $dir; ln -s $(basename $feature_transform) final.feature_transform )

###### WRITE DATA ######
feat-write ${feature_transform:+ --feature-transform=$feature_transform} ${use_gpu_id:+ --use-gpu-id=$use_gpu_id} --utts-per-file=500 "$feats" "$kaldi_dir/$dir/"

echo "Succeeded building data."

