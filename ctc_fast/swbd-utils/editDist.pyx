# cython: profile=False

# TODO Move to nn

from libc cimport math
import numpy as np
cimport numpy as np

def edit_distance(hyp, ref):
    '''
    Return minimum edit distance statistics as well
    as the operations which yield that distance
    http://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
    '''
    cdef np.int_t m = len(hyp)
    cdef np.int_t n = len(ref)
    # Initialized to 0
    cdef int ins = 0, dels = 0, subs = 0, eq = 0
    cdef np.int_t j, k

    # Typing the ndarray makes things so much faster not even funny
    cdef np.ndarray[np.int_t, ndim=2] D = np.empty((m+1, n+1), dtype=np.int)
    D[:, 0] = np.arange(m+1)
    D[0, :] = np.arange(n+1)

    for j in xrange(1, m+1):
        for k in xrange(1, n+1):
            if hyp[j-1] == ref[k-1]:
                D[j, k] = D[j-1, k-1]
            else:
                D[j, k] = 1 + min(D[j-1, k],    # deletion
                                  D[j, k-1],    # insertion
                                  D[j-1, k-1])  # substitution
    # Edit distance
    cdef int ed = D[m, n]

    cdef np.ndarray[np.int_t, ndim=1] errs_by_pos = np.zeros(m, dtype=np.int)

    # Compute # of insertions, deletions, substitutions
    j = m
    k = n
    hyp_corr = list()
    ref_corr = list()
    while j > 0 and k > 0:
        err_type_ind = min(j, k) - 1
        if hyp[j - 1] == ref[k - 1]:
            eq += 1
            hyp_corr.append(hyp[j-1])
            ref_corr.append(ref[k-1])
        elif D[j-1, k] == D[j, k] - 1:
            dels += 1
            errs_by_pos[j-1] += 1
            hyp_corr.append(hyp[j-1])
            ref_corr.append('<del>')
            k += 1
        elif D[j, k-1] == D[j, k] - 1:
            ins += 1
            errs_by_pos[j-1] += 1
            hyp_corr.append('<ins>')
            ref_corr.append(ref[k-1])
            j += 1
        elif D[j-1, k-1] == D[j, k] - 1:
            subs += 1
            errs_by_pos[j-1] += 1
            hyp_corr.append(hyp[j-1])
            ref_corr.append(ref[k-1])
        j -= 1
        k -= 1
    dels += j
    ins += k
    if m > 0:
        errs_by_pos[max(j-1, 0)] += j + k

    ref_corr += ['<del>'] * j
    while j > 0:
        hyp_corr.append(hyp[j-1])
        j -= 1
    hyp_corr += ['<ins>'] * k
    while k > 0:
        ref_corr.append(ref[k-1])
        k -= 1

    return ed, eq, ins, dels, subs, errs_by_pos, hyp_corr[::-1], ref_corr[::-1]
