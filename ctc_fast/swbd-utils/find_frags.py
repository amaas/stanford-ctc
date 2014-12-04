'''
Find and output utterances that contain fragments
'''

REF_UTT_FILE = '/deep/u/zxie/ctc_clm_transcripts/eval2000_text_ref'

with open(REF_UTT_FILE, 'r') as fin:
    lines = fin.read().strip().splitlines()
    for l in lines:
        utt_key, words = l.split(' ', 1)
        words = words.strip().split(' ')
        for word in words:
            if word.endswith('-'):
                print utt_key, word
