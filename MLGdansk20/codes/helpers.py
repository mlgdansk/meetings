
from math import sqrt
import numpy as np

# color coding of edge weights
#weightcolor = cm.ScalarMappable(
#    norm=colors.Normalize(vmin=0.0, vmax=1.0),
#    cmap=plt.get_cmap("gray_r")
#).to_rgba


def estimate_mu(samp):
    n = len(samp)
    return sum(samp)/n

def estimate_mu2d(samp):
    n = np.shape(samp)[0]
    return sum(samp[:,0])/n, sum(samp[:,1])/n

def estimate_sigma(samp, mu_est=None):
    if mu_est is None: mu = estimate_mu(samp)
    else: mu = mu_est
    n = len(samp)
    return sqrt(sum([ (x-mu)**2 for x in samp])) / (n-1)


class LinClassifier:

    def __init__(self, ABC, pexamp):
        A, B, C = ABC
        px, py = pexamp
        if A * px + B * py + C > 0:
            self.A = A ; self.B = B ; self.C = C
        else:
            self.A = -A ; self.B = -B ; self.C = -C
    
    def __call__(self, a, b=None):
        if b is None: x, y = a
        else: x, y = a, b
        return cmp(self.A * x + self.B * y + self.C, 0)

    def test(self, expect, n, distr):
        """test against a distribution and expected value"""
        assert expect == -1 or expect == 1
        cnt_right, cnt_wrong, cnt_undecided = 0, 0, 0
        for x,y in distr.rvs(n):
            res = self(x, y)
            if res == expect: cnt_right += 1
            elif res == 0: cnt_undecided += 1
            else: cnt_wrong += 1
        return cnt_right, cnt_wrong, cnt_undecided


