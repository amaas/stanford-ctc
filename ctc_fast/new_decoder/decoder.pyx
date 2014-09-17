# cython: profile=False

from libc cimport math
import numpy as np
cimport numpy as np
np.seterr(all='raise')
import collections
#import abc
# lm module. not required for all decoders
import kenlm

cdef class Hyp(object):
      """
      Container class for a single hypothesis
      """
      cdef double p_b
      cdef double p_nb
      cdef double p_c
      def __init__(self, pb, pnb, nc):
            self.p_b = pb
            self.p_nb = pnb
            self.n_c = nc
        
cdef class DecoderBase(object):
      """
      This is the base class for any decoder.
      It does not perform actual decoding (think of it as an abstract class)
      To make your decoder work, extend the class and implement the decode function
      """
      #__metaclass__ = abc.ABCMeta

       # TODO are static methods mixed with cython still fast?
      #@staticmethod
      cpdef double combine(self,double a,double b,double c=float('-inf')):
            cdef double psum = math.exp(a) + math.exp(b) + math.exp(c)
            if psum == 0.0:
                  return float('-inf')
            else:
                  return math.log(psum)             

      # character mapping objects. Cython requires declaring in advance
      cdef dict char_int_map
      cdef dict int_char_map
      def load_chars(self, charmap_file):
            """
            Loads a mapping of character -> int
            Stores mapping in self.char_int_map
            Stores int -> char mapping in self.int_char_map
            returns True if maps created successfully
            """
            with open(charmap_file) as fid:
                  self.char_int_map = dict(tuple(l.strip().split()) for l in fid.readlines())

            self.int_char_map = {}
            for k, v in self.char_int_map.iteritems():
                  self.char_int_map[k] = int(v)
                  self.int_char_map[int(v)] = k
            
            return True

      #cpdef seq_int_to_char(self, )
      #@abc.abstractmethod
      cpdef decode(self, double[::1,:] probs):
            """
            Child classes must implement the decode function
            Minimally the decode function takes a matrix of probabilities
            output by the network (characters vs time)
            returns the best hypothesis in characters
            """
            return None


cdef class ArgmaxDecoder(DecoderBase):
      """
      This is the simple argmax decoder. It doesn't need an LM
      It performs basic collapsing decoding
      """

      cpdef decode(self, double[::1,:] probs):
            """
            Takes matrix of probabilities and computes per-frame argmax
            Applies basic blank/duplicate collapsing
            returns the best hypothesis in characters
            Charmap must be loaded 
            """
            maxInd = np.argmax(probs, axis=0)
            pmInd = -1
            hyp = []
            # TODO is this the right way to score argmax decoding?
            hyp_score = 0.0
            for t in range(probs.shape[1]):
                hyp_score = hyp_score + probs[maxInd[t],t]
                if maxInd[t] != pmInd:
                    pmInd = maxInd[t]
                    if pmInd > 0:
                        hyp.append(self.int_char_map[pmInd])

            # collapsed hypothesis (this is our best guess)
            hyp =  ''.join(hyp)
            return hyp, hyp_score


cdef class BeamLMDecoder(DecoderBase):
      """
      Beam-search decoder with character LM
      """

      cdef object lm 
      def load_lm(self, lmfile):
            """
            Loads a language model from lmfile
            returns True if lm loading successful
            """
            self.lm = kenlm.LanguageModel(lmfile)
            return True

      def lm_score_final_char(self, prefix, query_char):
            """
            uses lm to score entire prefix
            returns only the log prob of final char
            """
            # convert prefix and query to actual text
            # TODO why is prefix a tuple?
            full_int = list(prefix) + [query_char]
            
            full_str = ' '.join([self.int_char_map[i] for i in full_int])
            full_scores = self.lm.full_scores(full_str)
            #words = ['<s>'] + full_str.split() + ['</s>']
            #TODO verify lm is not returning score for </s>
            prob_list = [x[0] for x in full_scores]
            # for i, (prob, length) in enumerate(prefix_scores):
            #    print words[i], length, prob
            return prob_list[-2]


      def decode(self, double[::1,:] probs,
                   unsigned int beam=40, double alpha=1.0, double beta=0.0):
            """
            Decoder with an LM
            returns the best hypothesis in characters
            Charmap must be loaded 
            """
            cdef unsigned int N = probs.shape[0]
            cdef unsigned int T = probs.shape[1]
            cdef unsigned int t, i
            cdef float v0, v1, v2, v3

            keyFn = lambda x: self.combine(x[1][0],x[1][1]) + beta * x[1][2]
            initFn = lambda : [float('-inf'),float('-inf'),0]

            # [prefix, [p_nb, p_b, node, |W|]]
            Hcurr = [[(),[float('-inf'),0.0,0]]]
            Hold = collections.defaultdict(initFn)

            # loop over time
            for t in xrange(T):
                  Hcurr = dict(Hcurr)
                  Hnext = collections.defaultdict(initFn)

                  for prefix,(v0,v1,numC) in Hcurr.iteritems():

                        valsP = Hnext[prefix]
                        valsP[1] = self.combine(v0+probs[0,t],v1+probs[0,t],valsP[1])
                        valsP[2] = numC
                        if len(prefix) > 0:
                              valsP[0] = self.combine(v0+probs[prefix[-1],t],valsP[0])

                        for i in xrange(1,N):
                              nprefix = tuple(list(prefix) + [i])
                              valsN = Hnext[nprefix]

                              # query the LM_SCORE_FINAL_CHAR for final char score
                              lm_prob = alpha * self.lm_score_final_char(prefix, i)
                              # lm_prob = alpha*lm_placeholder(i,prefix)
                              valsN[2] = numC + 1
                              if len(prefix)==0 or (len(prefix) > 0 and i != prefix[-1]):
                                    valsN[0] = self.combine(v0+probs[i,t]+lm_prob,v1+probs[i,t]+lm_prob,valsN[0])
                              else:
                                    valsN[0] = self.combine(v1+probs[i,t]+lm_prob,valsN[0])

                              if nprefix not in Hcurr:
                                    v2,v3,_ = Hold[nprefix]
                                    valsN[1] = self.combine(v2+probs[0,t],v3+probs[0,t],valsN[1])
                                    valsN[0] = self.combine(v2+probs[i,t],valsN[0])


                  Hold = Hnext
                  Hcurr = sorted(Hnext.iteritems(), key=keyFn, reverse=True)[:beam]


            hyp = ''.join([self.int_char_map[i] for i in Hcurr[0][0]])

            return hyp, keyFn(Hcurr[0])
            #return hyp
            #return list(Hcurr[0][0]),keyFn(Hcurr[0])
      
