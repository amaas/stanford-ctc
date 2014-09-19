source srilm_cfg.sh

order=$1
oldlm=$2
binlm=$3

$NGRAM -order $order -lm $oldlm -write-bin-lm $binlm
