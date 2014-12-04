For all, just need hyp.txt and run ./testSwbd.sh

---

For oovs and frags, first copy appropriate file to mergehyp.txt
(or first set hyp.txt and run ./testSwbd.sh)

Then run
    python score_frag_utts.py mergehyp.txt sclite_score/stm fragmergehyp.txt fragstm
or
    python score_oov_utts.py mergehyp.txt sclite_score/stm oovmergehyp.txt oovstm
TODO diff the two and update oov scoring

Then just run ./testFrag.sh or ./testOov.sh

For fragments, remember to turn off -F option in hubscr.pl
