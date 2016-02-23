import collections

kaldi_base = "/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/"

# Symbols
laugh = '[laughter]'
noise = '[noise]'
voc_noise = '[vocalized-noise]'
space = '[space]'

# Spell out integers
integers = ['zero','one','two','three','four','five','six','seven','eight','nine']

def unique_tokens():
    """
    Reads swbd transcripts and stores unique tokens.
    """
    with open('data/train/text','r') as fid:
	lines = [l.strip().split()[1:] for l in fid.readlines()]

    tokens = collections.defaultdict(int)
    for i,line in enumerate(lines):
	for l in line:
	    if l == laugh or l==noise or l==voc_noise:
		tokens[l] += 1
	    else:
		for t in list(l):
		    if t=='_':
			continue
		    try:
			int(t)
		    except ValueError:
			# Ignore integers
			tokens[t] += 1
	    tokens[space] += 1
	print "Parsed %d lines."%i

    fid = open('ctc-utils/chars.txt','w')
    for i,k in enumerate(tokens.keys()):
	fid.write(k+' '+str(i+1)+'\n')
    fid.close()

    return tokens

def tokenize(labels,file='data/train/text_ctc'):
    """
    Reads swbd transcripts and builds swbd k to list of 
    integer labels mapping.
    """

    with open(file,'r') as fid:
	lines = [l.strip().split() for l in fid.readlines()]
	data = dict((l[0],l[1:]) for l in lines)

    int_labels = [[labels[l] for l in list(i)] for i in integers]

    # for every utterance
    for k,line in data.iteritems():
	newline = []
	# for every word in transcription
	for i,word in enumerate(line):
	    # for [noise] etc
	    if word in labels.keys():
		newline.append(labels[word])
	    else:
		# for every char in word
		for j,char in enumerate(list(word)):
		    try: 
			newline.append(labels[char])
		    except KeyError:
			# Add spelled out integer followed by space
			newline += int_labels[int(char)]
			if j < len(list(word)) - 1:
			    newline.append(labels[space])


	    # Add a space inbetween every word
	    if i < len(line) -1:
		newline.append(labels[space])
	    
	data[k] = newline
    return data

def write_alis(utts,file=kaldi_base+'exp/train_ctc',numfiles=20):
    """
    Takes utterance to alignment mapping and splits it up 
    into alignment files according to file structure of 
    training set.
    """
    for f in range(1,numfiles+1):
	print "writing file %d..."%f
	with open(file+'/keys%d.txt'%f,'r') as fid:
	    keys = [l.strip().split()[0] for l in fid.readlines()]

	with open(file+'/alis%d.txt'%f,'w') as fid:
	    for k in keys:
		fid.write(k+" "+" ".join(utts[k])+'\n')

def load_labels():
    """
    Loads file with label to integer mapping. Use 
    unique_tokens to create file.
    """
    with open('ctc-utils/chars.txt','r') as fid:
	labels = dict(tuple(l.strip().split()) for l in fid.readlines())
    return labels

def compute_bigrams():
    """
    Compute bigrams with smoothing. Save in bigrams.bin.
    """
    import cPickle as pickle
    import numpy as np
    fid_bg = open(kaldi_base+'exp/train_ctc/bigrams.bin','w')
    labels = load_labels()
    numLabels = len(labels.keys())
    bigrams = np.ones((numLabels,numLabels))
    numfiles = 384

    for f in range(1,numfiles+1):
        print "Reading alis %d."%f
        with open('exp/train_ctc/alis%d.txt'%f,'r') as fid:
            alis = [l.strip().split()[1:] for l in fid.readlines()]
        for v in alis:
            for i,j in zip(v[1:],v[:-1]):
                bigrams[int(i)-1,int(j)-1] += 1
        
    bigrams = bigrams/np.sum(bigrams,axis=0)
    pickle.dump(bigrams,fid_bg)

    return bigrams

if __name__=='__main__':
#    unique_tokens()
    labelset = load_labels()
    data = [('train',384),('dev',20)]

    for name,num in data:
	utts = tokenize(labelset, file=kaldi_base+'data/%s/text_ctc'%name)
	write_alis(utts, file=kaldi_base+'exp/%s_ctc'%name,numfiles=num)



