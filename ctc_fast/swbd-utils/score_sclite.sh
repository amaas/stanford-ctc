#!/bin/bash

# begin configuration section.
# TODO HANDLE LMWT, KALDIROT, need some other files
. parse_options.sh || exit 1;

base=/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk
data=$base/egs/swbd/s5b/data/eval2000
dir=$3

hubscr=$KALDI_ROOT/tools/sctk-2.4.0/bin/hubscr.pl 
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $data/stm $data/glm $data/segments $data/reco2file_and_channel; 
do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval2000

mkdir -p $dir/scoring/log

utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
'>' $dir/score_LMWT/$name.ctm

# Remove some stuff we don't want to score, from the ctm.
for x in $dir/score_*/$name.ctm; do
cp $x $dir/tmpf;
cat $dir/tmpf | grep -i -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
  grep -i -v -E '<UNK>' > $x;
done

# Score the set...
$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
cp $data/stm $dir/score_LMWT/ '&&' \
$hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/score_LMWT/stm $dir/score_LMWT/${name}.ctm || exit 1;

# For eval2000 score the subsets

# Score only the, swbd part...
$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.swbd.LMWT.log \
grep -v '^en_' $data/stm '>' $dir/score_LMWT/stm.swbd '&&' \
grep -v '^en_' $dir/score_LMWT/${name}.ctm '>' $dir/score_LMWT/${name}.ctm.swbd '&&' \
$hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/score_LMWT/stm.swbd $dir/score_LMWT/${name}.ctm.swbd || exit 1;

# Score only the, callhome part...
$cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.callhm.LMWT.log \
grep -v '^sw_' $data/stm '>' $dir/score_LMWT/stm.callhm '&&' \
grep -v '^sw_' $dir/score_LMWT/${name}.ctm '>' $dir/score_LMWT/${name}.ctm.callhm '&&' \
$hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/score_LMWT/stm.callhm $dir/score_LMWT/${name}.ctm.callhm || exit 1;

exit 0
