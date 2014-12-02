'''
Find and output utterances that contain OOVs
'''

REF_UTT_FILE = '/deep/u/zxie/ctc_clm_transcripts/eval2000_text_ref'
WORD_LIST_FILE = '/deep/u/zxie/ctc_clm_transcripts/swbd_words.txt'

# Build up set of words in lexicon

lexicon = set()
with open(WORD_LIST_FILE, 'r') as fin:
    lines = fin.read().strip().splitlines()
    for l in lines:
        lexicon.add(l.split(' ')[0])

# Now go through utterances and determine which contain OOVs

with open(REF_UTT_FILE, 'r') as fin:
    lines = fin.read().strip().splitlines()
    for l in lines:
        utt_key, words = l.split(' ', 1)
        words = words.strip().split(' ')
        for word in words:
            if len(word) > 0 and word not in lexicon and word not in ['mhm']:
                print utt_key, word
