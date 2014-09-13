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
            for t in range(probs.shape[1]):
                if maxInd[t] != pmInd:
                    pmInd = maxInd[t]
                    if pmInd > 0:
                        hyp.append(self.int_char_map[pmInd])

            # collapsed hypothesis (this is our best guess)
            hyp =  ''.join(hyp)
            return hyp


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
            lm = kenlm.LanguageModel(lmfile)
            return True

      def lm_score_final_char(lm, chars, prefix):
            """
            uses lm to score entire prefix
            returns only the log prob of final char
            """
            # TODO
            #prefix_str = '<s> ' + ' '.join([chars[x] for x in prefix])
            #print prefix_str,
            #prefix_scores = lm.full_scores(prefix_str)
            #print prefix_scores[-1]


      cpdef decode(self, double[::1,:] probs):
            """
            XXX
            returns the best hypothesis in characters
            Charmap must be loaded 
            """
            maxInd = np.argmax(probs, axis=0)
            pmInd = -1
            hyp = []
            for t in range(probs.shape[1]):
                if maxInd[t] != pmInd:
                    pmInd = maxInd[t]
                    if pmInd > 0:
                        hyp.append(self.int_char_map[pmInd])

            # collapsed hypothesis (this is our best guess)
            hyp =  ''.join(hyp)
            return hyp
