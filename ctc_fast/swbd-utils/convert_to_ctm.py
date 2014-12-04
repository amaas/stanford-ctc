import sys

if len(sys.argv) > 1 and sys.argv[1] == 'oov':
    hyp_ctm = 'oovhyp.ctm'
    merge_file = 'oovmergehyp.txt'
elif len(sys.argv) > 1 and sys.argv[1] == 'frag':
    hyp_ctm = 'fraghyp.ctm'
    merge_file = 'fragmergehyp.txt'
else:
    hyp_ctm = 'hyp.ctm'
    merge_file = 'mergehyp.txt'

def load_hyp_txt(file=merge_file):
    with open(file, 'r') as fid:
        lines = fid.readlines()
    return lines

def write_ctm():
    fid = open(hyp_ctm, 'w')
    lines = load_hyp_txt()
    form = '%s %s %0.2f %0.2f %s\n'
    for l in lines:
        l = l.split(' ')
        assert len(l) > 1
        k, words = l[0], l[1:]
        words = [w.strip() for w in words]
        times = k.split('_')[2]
        start_time, end_time = [int(x) / 100.0 for x in times.split('-')]
        duration = end_time - start_time
        #dur_split = duration / max(float(len(words)), 1.0)
        if ('-a_' in k):
            channel = 'A'
        else:
            channel = 'B'
        t = 0
        for word in words:
            fid.write(form % (k[0:7], channel, start_time, duration, word))
            #fid.write(form % (k[0:7], channel, start_time, dur_split, word))
            #start_time += dur_split
            t += 1
    fid.close()

if __name__ == '__main__':
    write_ctm()
