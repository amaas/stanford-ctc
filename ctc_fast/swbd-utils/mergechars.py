
space = "[space]"
specials = set(["[noise]","[vocalized-noise]","[laughter]"])

with open('hyp.txt','r') as fid:
    lines = fid.readlines()

fid = open('mergehyp.txt','w')

def averageWords(text_f="/afs/cs.stanford.edu/u/awni/swbd/data/eval2000/text_ctc"):
    with open(text_f,'r') as fid:
        lines = [l.strip().split()[1:] for l in fid.readlines()]
        numUtts = float(len(lines))
        numWords = sum(len(l) for l in lines)
    return numWords/numUtts


numWords = 0.
numUtts = 0.
for l in lines:
    tokens = l.strip().split()
    k = tokens[0]
    words = []
    wordt = []
    for t in tokens[1:]:
        if t in specials:
            continue
        elif t != space:
            wordt.append(t)
        else:
            words.append(''.join(wordt))
            wordt = []
    words.append(''.join(wordt))
    numWords += len(words)
    numUtts += 1.
    fid.write(k.lower()+' '+' '.join(words)+'\n')

print "Avg Words Per Utt -- Hyp: %f  Ref: %f"%(numWords/numUtts,
        averageWords())

fid.close()

