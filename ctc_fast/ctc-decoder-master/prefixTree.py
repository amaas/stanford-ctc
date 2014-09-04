import imp
import sys
import collections
from decoder_config import LM_SOURCE, CHARMAP_PATH, SPACE,\
        SPECIALS_LIST, LM_PREFIX, LM_ARPA_FILE


def load_chars():
    with open(CHARMAP_PATH+'chars.txt') as fid:
        chars = dict(tuple(l.strip().split()) for l in fid.readlines())
    for k,v in chars.iteritems():
        chars[k] = int(v)
    return chars

def load_words():
    with open(CHARMAP_PATH+'wordlist') as fid:
        words = [l.strip() for l in fid.readlines()]
    print 'Loaded %d words' % len(words)
    return words

def scrub():
    words = load_words()
    chars = load_chars()
    fid = open(CHARMAP_PATH+'wordlist','w')
    specials = set(SPECIALS_LIST)
    for word in words:
        if word in specials:
            continue
        skip = False
        for t in word:
            # ignore words with bad symbols
            if t not in chars.keys():
                print word
                skip = True
                break
        if not skip:
            fid.write(word+'\n')
    fid.close()

class Node:
    def __init__(self):
        self.isPrefix = False
        self.isWord = False
        self.children = None

class PrefixTree:

    def __init__(self,chars,words,lm):
        specials = set(SPECIALS_LIST)
        self.path_count = 0
        self.lm = lm
        self.chars = chars
        self.root = Node()
        self.root.isPrefix = True
        self.space = self.chars[SPACE]
        self.nodeFn = lambda : Node()
        self.root.children = collections.defaultdict(self.nodeFn)
        for word in list(specials):
            node = self.root.children[self.chars[word]]
            node.isWord = True
            node.id = lm.get_word_id(word)

        count = 0
        for word in words:
            if (count % 10000) == 0:
                print ".",
                sys.stdout.flush()
            self.addPath(word,self.root,lm.get_word_id(word))
            count += 1


    def addPath(self,prefix,node,wordId):
        self.path_count += 1
        p,rest = prefix[0],prefix[1:]

        if node.children is None:
            node.children = collections.defaultdict(self.nodeFn)
        next = node.children[self.chars[p]]
        if len(rest)==0:
            next.isWord = True
            next.id = wordId
            return
        else:
            next.isPrefix = True
            self.addPath(rest,next,wordId)

def load_and_pickle_lm():
    import cPickle as pickle
    lmMod = imp.load_source('lm', LM_SOURCE)
    lm = lmMod.LM(arpafile=LM_ARPA_FILE)
    lm.to_file(os.path.split_exit(LM_ARPA_FILE)[0] + '.bin')

def load_lm():
    lmMod = imp.load_source('lm', LM_SOURCE)
    lm = lmMod.LM(arpafile=LM_ARPA_FILE)
    return lm

def loadPrefixTree():
    lm = load_lm()
    chars = load_chars()
    words = load_words()
    return PrefixTree(chars,words,lm)

if __name__=='__main__':
    pt = loadPrefixTree()
    print 'Added %d prefix paths' % pt.path_count
