from decoder_config import SPACE, SPECIALS_LIST


def preproc_transcript(transcript, num_lines=float('inf')):
    '''
    Remove utterance ids and lowercase
    '''
    transcript = transcript.strip()
    utts = list()
    l = 0
    for line in transcript.split('\n'):
        utt = line.split(' ', 1)[1].lower()
        utts.append(utt)
        l += 1
        if l >= num_lines:
            break
    return utts


def preproc_utts(utts):
    '''
    Convert raw text into character language model format
    Split characters up, add start and end symbols, replace spaces
    Avoid splitting up characters in specials tokens list
    Also remove hesitations and parentheses
    '''
    utt_sents = [utt.split(' ') for utt in utts]
    # FIXME PARAM specific to swbd eval2000
    utt_sents = [[word for word in utt if word != '(%hesitation)'] for utt in utt_sents]
    utt_sents = [[word[1:-1] if word.startswith('(') else word for word in utt] for utt in utt_sents]

    # Split up characters while preserving special characters and adding back spaces
    text = [[list(sent[k] + (' ' if k < len(sent) - 1 else ''))
            if sent[k] not in SPECIALS_LIST else [sent[k]]
            for k in xrange(len(sent))] for sent in utt_sents]
    # Flatten nested list
    text = [[c for w in s for c in w] for s in text]
    # Add start and end symbols and replace spaces with space token
    text = [['<s>'] + [c if c != ' ' else SPACE for c in utt] + ['</s>'] for utt in text]
    return text


if __name__ == '__main__':
    import sys
    text_file = sys.argv[1]
    out_file = sys.argv[2]

    text = open(text_file, 'r').read()
    text = preproc_utts(preproc_transcript(text))
    # Remove start and stop symbols
    text = [utt[1:-1] for utt in text]

    open(out_file, 'w').write('\n'.join([' '.join(utt) for utt in text]))
