
k=$1
#grep $k ~/swbd/data/train/text_ctc
grep $k ~/wsj/data/test_dev93/text_ctc
grep $k ~/ctc_fast/mergehyp.txt
