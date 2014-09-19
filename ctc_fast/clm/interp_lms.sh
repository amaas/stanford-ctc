source ./srilm_cfg.sh

order=$1
lm1=$2
lm2=$3
outlm=$4
LAMBDAS=(0.5)

# NOTE Pass in 1 less lambda value than # of lms
# and last lambda (mix-lambda1) = 1 minus the rest

$NGRAM -order $order\
       -lm $lm1\
       -mix-lm $lm2\
       -lambda ${LAMBDAS[0]}\
       -write-lm $outlm
