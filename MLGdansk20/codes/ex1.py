#! /usr/bin/env python

import argparse

import random
import numpy as np
import scipy as sp
import scipy.stats as stats
np.random.seed(1)

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from math import sqrt

from helpers import *
from plt_line import plt_line, _sl_gen


argparser = argparse.ArgumentParser()
argparser.add_argument("--pdf", action="store_true",
                       help="output graphs to ex1.pdf")
args = argparser.parse_args()
if args.pdf:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages("ex1.pdf")

    
### symmetric gaussian distributions

sigma1 = 0.5
sigma2 = 1.0

mean1 = [-1.0, 0.0]; cov1 = [[sigma1**2, 0.0], [0.0, sigma1**2]]
NV1 = stats.multivariate_normal(mean1, cov1)
samp1 = NV1.rvs(50)
samp1_mu = estimate_mu2d(samp1)
samp1_sigma = estimate_sigma(samp1[:,0], samp1_mu[0])

mean2 = [1.0, 0.0]; cov2 = [[sigma2**2, 0.0], [0.0, sigma2**2]]
NV2 = stats.multivariate_normal(mean2, cov2)
samp2 = NV2.rvs(50)
samp2_mu = estimate_mu2d(samp2)
samp2_sigma = estimate_sigma(samp2[:,0], samp2_mu[0])

#fig, ax = plt.subplots(1, 1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(*zip(*samp1), c="blue", s=50, label="mean=(-1,0), sigma=0.5")
ax.scatter(*zip(*samp2), c="green", s=50, label="mean=(1,0), sigma=1.0")
ax.scatter(*mean1, c="blue", s=500, marker="+")
ax.scatter(*mean2, c="green", s=500, marker="+")
ax.scatter(*samp1_mu, c="blue", s=500, marker="o", alpha=0.4)
ax.scatter(*samp2_mu, c="green", s=500, marker="o", alpha=0.4)
ax.legend(bbox_to_anchor=(1.4, 1.05))
fig.subplots_adjust(right=0.7)

xlim=[-2.0, 4.0]
ylim=[-3.0, 3.0]
ax.set(xlim=xlim)
ax.set(ylim=ylim)
plt.draw()
if args.pdf: plt.savefig(pdf, format="pdf")
else: plt.waitforbuttonpress()  #pause(10)


### naive linear classifier

m1x = samp1_mu[0] ; m1y = samp1_mu[1]
m2x = samp2_mu[0] ; m2y = samp2_mu[1]
P = (0.5*(m1x+m2x), 0.5*(m1y+m2y))
N = (m2x-m1x, m2y-m1y)
ABC = _sl_gen(P, N=N)
naive_lcr = LinClassifier(ABC, samp2_mu)
r1, w1, u1 = naive_lcr.test(-1, 1000, NV1)
r2, w2, u2 = naive_lcr.test(1, 1000, NV2)
acc = (r1+r2)/20.0
print "Naive classifier testing:"
print "left:", r1, w1, u1, "right:", r2, w2, u2, "==> % acc:", acc
plt_line(ax, xlim, ylim, P, N=N, c="red",
         label="naive classifier\naccuracy=%.2f%%\n(%d/%d, %d/%d)"
         % (acc, r1, w1, r2, w2))
ax.legend(bbox_to_anchor=(1.4, 1.05))

plt.draw()
if args.pdf: plt.savefig(pdf, format="pdf")
else: plt.waitforbuttonpress()


### with Mahalanobis distance correction

# we take a simplified approach here and don't base on estimatin the variance
# but take the known values. So My = 0 and we compute Mx based on the following
# equation:
#  (Mx+1)/sigma1 == (1-Mx)/sigma2,
#  Mx(1/sigma1 + 1/sigma2) == 1/sigma2 - 1/sigma1 
Mx = (sigma1 - sigma2) / (sigma1 + sigma2)
M = (Mx, 0.0)
N = (1.0, 0.0)
ABC = _sl_gen(M, N=N)
mahal_lcr = LinClassifier(ABC, samp2_mu)
r1, w1, u1 = mahal_lcr.test(-1, 1000, NV1)
r2, w2, u2 = mahal_lcr.test(1, 1000, NV2)
acc = (r1+r2)/20.0
print "Naive classifier with Mahalanobis correction testing:"
print "left:", r1, w1, u1, "right:", r2, w2, u2, "==> % acc:", acc
plt_line(ax, xlim, ylim, M, N=N, c="orange",
         label="with Mahalanobis\ncorrection\naccuracy=%.2f%%\n(%d/%d, %d/%d)"
         % (acc, r1, w1, r2, w2))
ax.legend(bbox_to_anchor=(1.4, 1.05))

plt.draw()
if args.pdf: plt.savefig(pdf, format="pdf")
else: plt.waitforbuttonpress()


if args.pdf: pdf.close()
