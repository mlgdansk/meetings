#! /usr/bin/env python

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy, sklearn
numpy.random.seed(314)
from sklearn import datasets, svm
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def trans2d(tobj, arr):
    arr2d = tobj.transform(arr)
    return arr2d[:,0], arr2d[:,1]

import sys, argparse
scriptnm = sys.argv[0].split(".py")[0]
pdfnm = scriptnm + ".pdf"
argparser = argparse.ArgumentParser()
argparser.add_argument("--pdf", action="store_true",
                       help="output graphs to "+pdfnm)
args = argparser.parse_args()
if args.pdf:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(pdfnm)

iris = datasets.load_iris()

# Setosa, Versicolor and Virginica
setosa_data = iris.data[0:50,:]
versic_data = iris.data[50:100,:]
virgin_data = iris.data[100:150,:]
numpy.random.shuffle(setosa_data)
numpy.random.shuffle(versic_data)
numpy.random.shuffle(virgin_data)
setosa_data_train = setosa_data[0:30,:]
versic_data_train = versic_data[0:30,:]
virgin_data_train = virgin_data[0:30,:]
setosa_data_test = setosa_data[30:50,:]
versic_data_test = versic_data[30:50,:]
virgin_data_test = virgin_data[30:50,:]

# plotting
c1, c2, c3 = "blue", "red", "green"
s = 40

# do PCA to visualize training data
subp = 1
plt.subplot(2, 2, subp)
pca2d = PCA(n_components=2)
pca2d.fit(numpy.concatenate((setosa_data_train, versic_data_train, virgin_data_train)))
data2d = pca2d.transform(numpy.concatenate((setosa_data_train, versic_data_train, virgin_data_train)))
plt.title("Iris data, linear 2d PCA")
plt.scatter(*(trans2d(pca2d, setosa_data_train)), c=c1, s=s, label="Setosa train")
plt.scatter(*(trans2d(pca2d, setosa_data_test)), c=c1, s=s, marker="x", label="Setosa test")
plt.scatter(*(trans2d(pca2d, versic_data_train)), c=c2, s=s, label="Versicolor train")
plt.scatter(*(trans2d(pca2d, versic_data_test)), c=c2, s=s, marker="x", label="Versicolor test")
plt.scatter(*(trans2d(pca2d, virgin_data_train)), c=c3, s=s, label="Virginica train")
plt.scatter(*(trans2d(pca2d, virgin_data_test)), c=c3, s=s, marker="x", label="Virginica test")

# and kernel PCA with gaussian kernel
for gamma in [1.0, 5.0, 0.2]:
    subp += 1
    plt.subplot(2, 2, subp)
    kpca2d = KernelPCA(kernel="rbf", gamma=gamma, fit_inverse_transform=True, n_components=2)
    kpca2d.fit(numpy.concatenate((setosa_data_train, versic_data_train, virgin_data_train)))
    data2d = kpca2d.transform(numpy.concatenate((setosa_data_train, versic_data_train, virgin_data_train)))
    plt.title("gaussian KPCA, gamma=%.1f" % gamma)
    plt.scatter(*(trans2d(kpca2d, setosa_data_train)), c=c1, s=s, label="Setosa train")
    plt.scatter(*(trans2d(kpca2d, setosa_data_test)), c=c1, s=s, marker="x", label="Setosa test")
    plt.scatter(*(trans2d(kpca2d, versic_data_train)), c=c2, s=s, label="Versicolor train")
    plt.scatter(*(trans2d(kpca2d, versic_data_test)), c=c2, s=s, marker="x", label="Versicolor test")
    plt.scatter(*(trans2d(kpca2d, virgin_data_train)), c=c3, s=s, label="Virginica train")
    plt.scatter(*(trans2d(kpca2d, virgin_data_test)), c=c3, s=s, marker="x", label="Virginica test")


if args.pdf:
    plt.draw()
    plt.savefig(pdf, format="pdf")
else:
    plt.show()
plt.gcf().clear()

if args.pdf: pdf.close()
