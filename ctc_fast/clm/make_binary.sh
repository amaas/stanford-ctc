# verbose
set -x

ngram=/afs/cs.stanford.edu/u/zxie/libs/srilm/bin/i686-m64/ngram
order=$1
oldlm=$2
binlm=$3

$ngram -order $order -lm $oldlm -write-bin-lm $binlm
