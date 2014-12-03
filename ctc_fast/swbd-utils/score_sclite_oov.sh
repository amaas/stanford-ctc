#!/bin/bash

# begin configuration section.
# TODO Handle LMWT properly

# Verbose
set -x

KALDI_ROOT=/deep/group/speech/zxie/kaldi-stanford/kaldi-trunk
data=$KALDI_ROOT/egs/swbd/s5b/data/eval2000
dir=$1
ctmFile=$2
cmd=/deep/group/speech/zxie/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/utils/run.pl

hubscr=$KALDI_ROOT/tools/sctk-2.4.0/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $data/stm $data/glm $data/segments $data/reco2file_and_channel;
do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval2000

mkdir -p $dir/log

#utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
#'>' $dir/score_LMWT/$name.ctm

# Remove some stuff we don't want to score, from the ctm.
#for x in $dir/score_*/$name.ctm; do
cp $ctmFile $dir/tmpf;
cat $dir/tmpf | grep -i -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
  grep -i -v -E '<UNK>' > $dir/${name}.ctm;
#done

# Score the set...
#$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
$cmd $dir/log/score.log \
cp oovstm $dir/stm '&&' \
$hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/stm $dir/${name}.ctm || exit 1;

# For eval2000 score the subsets

# Score only the, swbd part...
#$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.swbd.LMWT.log \
$cmd $dir/log/score.swbd.log \
grep -v '^en_' $dir/stm '>' $dir/stm.swbd '&&' \
grep -v '^en_' $dir/${name}.ctm '>' $dir/${name}.ctm.swbd '&&' \
$hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/stm.swbd $dir/${name}.ctm.swbd || exit 1;

# Score only the, callhome part...
#$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.callhm.LMWT.log \
$cmd $dir/log/score.callhm.log \
grep -v '^sw_' $dir/stm '>' $dir/stm.callhm '&&' \
grep -v '^sw_' $dir/${name}.ctm '>' $dir/${name}.ctm.callhm '&&' \
$hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/stm.callhm $dir/${name}.ctm.callhm || exit 1;

exit 0
