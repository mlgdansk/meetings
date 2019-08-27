#! /usr/bin/env python

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy, sklearn
numpy.random.seed(314)
from sklearn import datasets, svm
from sklearn.decomposition import PCA
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
pca2d = PCA(n_components=2)
pca2d.fit(numpy.concatenate((setosa_data_train, versic_data_train, virgin_data_train)))
data2d = pca2d.transform(numpy.concatenate((setosa_data_train, versic_data_train, virgin_data_train)))
plt.title("Iris data, PCA 2d transformation")
plt.scatter(*(trans2d(pca2d, setosa_data_train)), c=c1, s=s, label="Setosa train")
plt.scatter(*(trans2d(pca2d, setosa_data_test)), c=c1, s=s, marker="x", label="Setosa test")
plt.scatter(*(trans2d(pca2d, versic_data_train)), c=c2, s=s, label="Versicolor train")
plt.scatter(*(trans2d(pca2d, versic_data_test)), c=c2, s=s, marker="x", label="Versicolor test")
plt.scatter(*(trans2d(pca2d, virgin_data_train)), c=c3, s=s, label="Virginica train")
plt.scatter(*(trans2d(pca2d, virgin_data_test)), c=c3, s=s, marker="x", label="Virginica test")
plt.legend(bbox_to_anchor=(1.12, 1.01))  #1.4,1.05
if args.pdf:
    plt.draw()
    plt.savefig(pdf, format="pdf")
else:
    plt.show()
plt.gcf().clear()

# mesh to visualise classification regions
all2d = pca2d.transform(iris.data)
x_min, x_max = all2d[:,0].min() - 0.3, all2d[:,0].max() + 0.3
y_min, y_max = all2d[:,1].min() - 0.3, all2d[:,1].max() + 0.3
h = 0.02
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
xxf = xx.flatten()
yyf = yy.flatten()
grid = zip(xxf, yyf)
invgrid = pca2d.inverse_transform(grid)

# LDA classifier
fisher= LDA(n_components=2, solver="eigen")
fisher.fit(numpy.concatenate((versic_data_train, virgin_data_train)),
           [-1] * 30 + [1] * 30)
# score on training and test data
train_acc = fisher.score(numpy.concatenate((versic_data_train, virgin_data_train)),
                          [-1] * 30 + [1] * 30)
test_acc = fisher.score(numpy.concatenate((versic_data_test, virgin_data_test)),
                         [-1] * 20 + [1] * 20)
train_pacc = int(round(train_acc * 100.0))
test_pacc = int(round(test_acc * 100.0))
print "Fisher classifier Versicolor/Virginica, train/test accuracy (%%):", train_pacc, test_pacc
# scoring on the grid
ongrid = fisher.predict(invgrid)
plt.title("Fisher classifier Versicolor/Virginica, train/test accuracy (%%): %d/%d" % (train_pacc, test_pacc))
plt.scatter(*(trans2d(pca2d, setosa_data_train)), c=c1, s=s)
plt.scatter(*(trans2d(pca2d, setosa_data_test)), c=c1, s=s, marker="x")
plt.scatter(*(trans2d(pca2d, versic_data_train)), c=c2, s=s)
plt.scatter(*(trans2d(pca2d, versic_data_test)), c=c2, s=s, marker="x")
plt.scatter(*(trans2d(pca2d, virgin_data_train)), c=c3, s=s)
plt.scatter(*(trans2d(pca2d, virgin_data_test)), c=c3, s=s, marker="x")
plt.contourf(xx, yy, ongrid.reshape(xx.shape), cmap=plt.cm.RdYlGn, alpha=0.3)
if args.pdf:
    plt.draw()
    plt.savefig(pdf, format="pdf")
else:
    plt.show()
plt.gcf().clear()

# linear SVM classifier Versicolor/Virginica
C = 1.0  # regularization weight
svc_lin = svm.SVC(kernel="linear", C=C)
svc_lin.fit(numpy.concatenate((versic_data_train, virgin_data_train)),
            [-1] * 30 + [1] * 30)
# score on training and test data
train_acc = svc_lin.score(numpy.concatenate((versic_data_train, virgin_data_train)),
                          [-1] * 30 + [1] * 30)
test_acc = svc_lin.score(numpy.concatenate((versic_data_test, virgin_data_test)),
                         [-1] * 20 + [1] * 20)
train_pacc = int(round(train_acc * 100.0))
test_pacc = int(round(test_acc * 100.0))
print "Linear SVC Versicolor/Virginica, train/test accuracy (%%):", train_pacc, test_pacc
# scoring on the grid
ongrid = svc_lin.predict(invgrid)
plt.title("Linear SVC Versicolor/Virginica, train/test accuracy (%%): %d/%d" % (train_pacc, test_pacc))
plt.scatter(*(trans2d(pca2d, setosa_data_train)), c=c1, s=s)
plt.scatter(*(trans2d(pca2d, setosa_data_test)), c=c1, s=s, marker="x")
plt.scatter(*(trans2d(pca2d, versic_data_train)), c=c2, s=s)
plt.scatter(*(trans2d(pca2d, versic_data_test)), c=c2, s=s, marker="x")
plt.scatter(*(trans2d(pca2d, virgin_data_train)), c=c3, s=s)
plt.scatter(*(trans2d(pca2d, virgin_data_test)), c=c3, s=s, marker="x")
plt.contourf(xx, yy, ongrid.reshape(xx.shape), cmap=plt.cm.RdYlGn, alpha=0.3)
if args.pdf:
    plt.draw()
    plt.savefig(pdf, format="pdf")
else:
    plt.show()
plt.gcf().clear()

# cubic SVM classifier Versicolor/Virginica
C = 1.0  # regularization weight
svc_cub = svm.SVC(kernel="poly", degree=3, C=C)
svc_cub.fit(numpy.concatenate((versic_data_train, virgin_data_train)),
            [-1] * 30 + [1] * 30)
# score on training and test data
train_acc = svc_cub.score(numpy.concatenate((versic_data_train, virgin_data_train)),
                          [-1] * 30 + [1] * 30)
test_acc = svc_cub.score(numpy.concatenate((versic_data_test, virgin_data_test)),
                         [-1] * 20 + [1] * 20)
train_pacc = int(round(train_acc * 100.0))
test_pacc = int(round(test_acc * 100.0))
print "Cubic SVC Versicolor/Virginica, train/test accuracy (%%):", train_pacc, test_pacc
# scoring on the grid
ongrid = svc_cub.predict(invgrid)
plt.title("Cubic SVC Versicolor/Virginica, train/test accuracy (%%): %d/%d" % (train_pacc, test_pacc))
plt.scatter(*(trans2d(pca2d, setosa_data_train)), c=c1, s=s)
plt.scatter(*(trans2d(pca2d, setosa_data_test)), c=c1, s=s, marker="x")
plt.scatter(*(trans2d(pca2d, versic_data_train)), c=c2, s=s)
plt.scatter(*(trans2d(pca2d, versic_data_test)), c=c2, s=s, marker="x")
plt.scatter(*(trans2d(pca2d, virgin_data_train)), c=c3, s=s)
plt.scatter(*(trans2d(pca2d, virgin_data_test)), c=c3, s=s, marker="x")
plt.contourf(xx, yy, ongrid.reshape(xx.shape), cmap=plt.cm.RdYlGn, alpha=0.3)
if args.pdf:
    plt.savefig(pdf, format="pdf")
else:
    plt.show()
plt.gcf().clear()

# gaussian SVM classifier Versicolor/Virginica
C = 1.0  # regularization weight
svc_rbf = svm.SVC(kernel="rbf", gamma=1.0, C=C)
svc_rbf.fit(numpy.concatenate((versic_data_train, virgin_data_train)),
            [-1] * 30 + [1] * 30)
# score on training and test data
train_acc = svc_rbf.score(numpy.concatenate((versic_data_train, virgin_data_train)),
                          [-1] * 30 + [1] * 30)
test_acc = svc_rbf.score(numpy.concatenate((versic_data_test, virgin_data_test)),
                                 [-1] * 20 + [1] * 20)
train_pacc = int(round(train_acc * 100.0))
test_pacc = int(round(test_acc * 100.0))
print "Gaussian SVC Versicolor/Virginica, train/test accuracy:", train_pacc, test_pacc
# scoring on the grid
ongrid = svc_rbf.predict(invgrid)
plt.title("Gaussian SVC Versicolor/Virginica, train/test accuracy (%%): %d/%d" % (train_pacc, test_pacc))
plt.scatter(*(trans2d(pca2d, setosa_data_train)), c=c1, s=s)
plt.scatter(*(trans2d(pca2d, setosa_data_test)), c=c1, s=s, marker="x")
plt.scatter(*(trans2d(pca2d, versic_data_train)), c=c2, s=s)
plt.scatter(*(trans2d(pca2d, versic_data_test)), c=c2, s=s, marker="x")
plt.scatter(*(trans2d(pca2d, virgin_data_train)), c=c3, s=s)
plt.scatter(*(trans2d(pca2d, virgin_data_test)), c=c3, s=s, marker="x")
plt.contourf(xx, yy, ongrid.reshape(xx.shape), cmap=plt.cm.RdYlGn, alpha=0.3)
if args.pdf:
    plt.savefig(pdf, format="pdf")
else:
    plt.show()
plt.gcf().clear()

# square SVM classifier Versicolor/Virginica
C = 1.0  # regularization weight
svc_sq = svm.SVC(kernel="poly", degree=2, C=C)
svc_sq.fit(numpy.concatenate((versic_data_train, setosa_data_train, virgin_data_train)),
           [-1] * 30 + [1] * 60)
# score on training and test data
train_acc = svc_sq.score(numpy.concatenate((versic_data_train, setosa_data_train, virgin_data_train)),
                         [-1] * 30 + [1] * 60)
test_acc = svc_sq.score(numpy.concatenate((versic_data_test, setosa_data_test, virgin_data_test)),
                        [-1] * 20 + [1] * 40)
train_pacc = int(round(train_acc * 100.0))
test_pacc = int(round(test_acc * 100.0))
print "Square SVC Versicolor/Setosa+Virginica, train/test accuracy:", train_pacc, test_pacc
# scoring on the grid
ongrid = svc_sq.predict(invgrid)
plt.title("Square SVC Versicolor/Setosa+Virginica, accuracy (%%): %d/%d" % (train_pacc, test_pacc))
plt.scatter(*(trans2d(pca2d, setosa_data_train)), c=c3, s=s)
plt.scatter(*(trans2d(pca2d, setosa_data_test)), c=c3, s=s, marker="x")
plt.scatter(*(trans2d(pca2d, versic_data_train)), c=c2, s=s)
plt.scatter(*(trans2d(pca2d, versic_data_test)), c=c2, s=s, marker="x")
plt.scatter(*(trans2d(pca2d, virgin_data_train)), c=c3, s=s)
plt.scatter(*(trans2d(pca2d, virgin_data_test)), c=c3, s=s, marker="x")
plt.contourf(xx, yy, ongrid.reshape(xx.shape), cmap=plt.cm.RdYlGn, alpha=0.3)
if args.pdf:
    plt.draw()
    plt.savefig(pdf, format="pdf")
else:
    plt.show()
plt.gcf().clear()


if args.pdf: pdf.close()
