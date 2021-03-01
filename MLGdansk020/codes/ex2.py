#! /usr/bin/env python

import sys, argparse

import random
import numpy as np
import scipy as sp
import scipy.stats as stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC as LSVC
np.random.seed(1)

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from math import sqrt

from helpers import *
from plt_line import plt_line, general_sl_coeffs

scriptnm = sys.argv[0].split(".py")[0]
pdfnm = scriptnm + ".pdf"

argparser = argparse.ArgumentParser()
argparser.add_argument("--pdf", action="store_true",
                       help="output graphs to "+pdfnm)
args = argparser.parse_args()
if args.pdf:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(pdfnm)

    
### symmetric gaussian distributions

n = 50  #10
sigma1x, sigma1y = 0.5, 2.0
sigma2x, sigma2y = 0.5, 2.0
#n = 20
#sigma1x, sigma1y = 0.35, 1.8
#sigma2x, sigma2y = 0.35, 1.8

mean1 = [-1.0, -1.0]; cov1 = [[sigma1x**2, 0.0], [0.0, sigma1y**2]]
NV1 = stats.multivariate_normal(mean1, cov1)
samp1 = NV1.rvs(n)
samp1_mu = estimate_mu2d(samp1)
samp1_sigma = estimate_sigma(samp1[:,0], samp1_mu[0])

mean2 = [1.0, 1.0]; cov2 = [[sigma2x**2, 0.0], [0.0, sigma2y**2]]
NV2 = stats.multivariate_normal(mean2, cov2)
samp2 = NV2.rvs(n)
samp2_mu = estimate_mu2d(samp2)
samp2_sigma = estimate_sigma(samp2[:,0], samp2_mu[0])

print samp1.shape, samp2.shape

#fig, ax = plt.subplots(1, 1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(*zip(*samp1), c="blue", s=50, label="mean=(-1,-1),\nsigma x/y=0.5/2.0")
ax.scatter(*zip(*samp2), c="green", s=50, label="mean=(1,1),\nsigma x/y=0.5/2.0")
ax.scatter(*mean1, c="blue", s=500, marker="+")
ax.scatter(*mean2, c="green", s=500, marker="+")
ax.scatter(*samp1_mu, c="blue", s=500, marker="o", alpha=0.4)
ax.scatter(*samp2_mu, c="green", s=500, marker="o", alpha=0.4)
ax.legend(bbox_to_anchor=(1.4, 1.05))
fig.subplots_adjust(right=0.7)

xlim=[-4.0, 4.0]
ylim=[-4.0, 4.0]
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
ABC = general_sl_coeffs(P, N=N)
naive_lcr = LinClassifier(ABC, samp2_mu)
r1, w1, u1 = naive_lcr.test(-1, 1000, NV1)
r2, w2, u2 = naive_lcr.test(1, 1000, NV2)
acc = (r1+r2)/20.0
print "Naive classifier testing:"
print "left:", r1, w1, u1, "right:", r2, w2, u2, "==> % acc:", acc
plt_line(ax, xlim, ylim, ABC, c="red", lw=2,
         label="naive classifier\naccuracy=%.2f%%\n(%d/%d, %d/%d)"
         % (acc, r1, w1, r2, w2))
ax.legend(bbox_to_anchor=(1.4, 1.05))

plt.draw()
if args.pdf: plt.savefig(pdf, format="pdf")
else: plt.waitforbuttonpress()


### Fisher classifier

fisher= LDA(n_components=2, solver="eigen")
tr_data = np.concatenate((samp1, samp2))
tr_objective = [-1]*n + [1]*n
print tr_data.shape, len(tr_objective)
fisher.fit(tr_data, tr_objective)

print fisher.coef_
print fisher.intercept_
(A, B), = fisher.coef_
C, = fisher.intercept_
print A, B, C
fisher_lcr = LinClassifier((A,B,C), samp2_mu)
r1, w1, u1 = fisher_lcr.test(-1, 1000, NV1)
r2, w2, u2 = fisher_lcr.test(1, 1000, NV2)
acc = (r1+r2)/20.0
print "Fisher classifier testing:"
print "left:", r1, w1, u1, "right:", r2, w2, u2, "==> % acc:", acc
plt_line(ax, xlim, ylim, (A,B,C), c="orange", lw=2,
         label="Fisher classifier\naccuracy=%.2f%%\n(%d/%d, %d/%d)"
         % (acc, r1, w1, r2, w2))
ax.legend(bbox_to_anchor=(1.4, 1.05))

plt.draw()
if args.pdf: plt.savefig(pdf, format="pdf")
else: plt.waitforbuttonpress()


### linear SVM classifier

lsvc= LSVC()
lsvc.fit(tr_data, tr_objective)
print lsvc.coef_
print lsvc.intercept_
(A, B), = lsvc.coef_
C, = lsvc.intercept_
print A, B, C
lsvc_lcr = LinClassifier((A,B,C), samp2_mu)
r1, w1, u1 = lsvc_lcr.test(-1, 1000, NV1)
r2, w2, u2 = lsvc_lcr.test(1, 1000, NV2)
acc = (r1+r2)/20.0
print "linear SVM classifier testing:"
print "left:", r1, w1, u1, "right:", r2, w2, u2, "==> % acc:", acc
plt_line(ax, xlim, ylim, (A,B,C), c="brown", lw=2, #alpha=0.8,
         label="linear SVM classifier\naccuracy=%.2f%%\n(%d/%d, %d/%d)"
         % (acc, r1, w1, r2, w2))
ax.legend(bbox_to_anchor=(1.4, 1.05))

plt.draw()
if args.pdf: plt.savefig(pdf, format="pdf")
else: plt.waitforbuttonpress()


if args.pdf: pdf.close()
