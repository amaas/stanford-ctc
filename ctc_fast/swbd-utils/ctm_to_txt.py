from collections import defaultdict

'''
Convert .ctm files to .txt files so can easily compare transcripts
since HMM-GMM baseline uses lattices and we only have the .ctm file
for scoring
'''

CTM_FILE = '/deep/group/speech/zxie/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/tri4b/decode_eval2000_sw1_fsh_tgpr/score_15/eval2000.ctm'

# Need this to determine which words from ctm file belong to which utterance
REF_UTT_FILE = '/deep/u/zxie/ctc_clm_transcripts/eval2000_text_ref'

# Build up utterances and timings
utt_keys = list()
utt_times_dict = defaultdict(list)
with open(REF_UTT_FILE, 'r') as fin:
    utt_lines = fin.read().strip().splitlines()
for l in utt_lines:
    utt_key_and_times, utt = l.split(' ', 1)
    parts = utt_key_and_times.split('_')
    utt_key, utt_times = '_'.join(parts[0:2]), parts[2]
    start_time, end_time = [int(t) for t in utt_times.split('-')]
    if len(utt_keys) == 0 or utt_keys[-1] != utt_key:
        utt_keys.append(utt_key)
    utt_times_dict[utt_key].append((start_time, end_time))

# Build up the words for each utterance + time range from .ctm file
utt_times_words_dict = defaultdict(lambda: defaultdict(list))
with open(CTM_FILE, 'r') as fin:
    ctm_lines = fin.read().strip().splitlines()
for l in ctm_lines:
    utt_key_partial, utt_channel, t_start, t_dur, word = l.split(' ')
    utt_key = utt_key_partial + '-' + utt_channel.lower()
    start_time = int(float(t_start) * 100)
    end_time = int((float(t_start) + float(t_dur)) * 100)
    # Find which of the reference key and times the word belongs to
    for t in utt_times_dict[utt_key]:
        if start_time >= t[0] and end_time <= t[1]:
            utt_times_words_dict[utt_key][t].append(word)


for utt_key in utt_keys:
    for t in utt_times_dict[utt_key]:
        print '%s_%06d-%06d %s' % (utt_key, t[0], t[1], ' '.join(utt_times_words_dict[utt_key][t]))
