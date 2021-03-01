from itertools import chain

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.init import strong_ent_layers_uniform
from pennylane.optimize import NesterovMomentumOptimizer

np.random.seed(42)

# create the dataset
X, y = datasets.make_moons(100, noise=0.1, random_state=42)

# shuffle the data
X, y = shuffle(X, y, random_state=42)


# normalize data
X = minmax_scale(X, feature_range=(0, np.pi))

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

# number of qubits is equal to the number of features
n_qubits = X.shape[1] 

# quantum device handle
dev = qml.device("default.qubit", wires=n_qubits)

# quantum circuit
@qml.qnode(dev)
def circuit(weights, x=None):
    AngleEmbedding(x, wires = range(n_qubits))
    StronglyEntanglingLayers(weights, wires = range(n_qubits))
    return qml.expval(qml.PauliZ(0))

def cost(theta, X, expectations):
    e_predicted = \
        np.array([circuit(theta, x=x) for x in X])

    loss = np.mean((e_predicted - expectations)**2)
    return loss

# helper function: returns 1 if x>0 else 0
def sgn(x):
    return (x>0)*1

# number of quantum layers
n_layers = 3

# convert classes to expectations: 0 to -1, 1 to +1
e_train = np.empty_like(y_train)
e_train[y_train == 0] = -1
e_train[y_train == 1] = +1

# select learning batch size
batch_size = 5

# calculate number of batches
batches = len(X_train) // batch_size

# select number of epochs
n_epochs = 5

# draw random quantum node weights
theta = strong_ent_layers_uniform(n_layers, n_qubits, seed=15)

# train the variational classifier

# start of main learning loop
# build the optimizer object
pennylane_opt = NesterovMomentumOptimizer()

log = []
# split training data into batches
X_batches = np.array_split(np.arange(len(X_train)), batches)
for it, batch_index in enumerate(chain(*(n_epochs * [X_batches]))):
    # Update the weights by one optimizer step
    batch_cost = \
        lambda t: cost(t, X_train[batch_index], e_train[batch_index])
    theta = pennylane_opt.step(batch_cost, theta)
    log.append({"theta":theta})
# end of learning loop

# convert scores to classes
scores = np.array([circuit(theta, x=x) for x in X_test])
y_pred = sgn(scores)

print(metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))



def plot(X, y, log, name="", density = 23):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # data = [(i, step["batch_cost"]) for i, step in enumerate(log)]
    # data = list(zip(*data))
    # plt.figure(figsize=(11, 11))
    # plt.plot(*data)
    # plt.title(f"Learning curve")
    # plt.xlabel("Learning step")
    # plt.ylabel("Batch loss")
    # plt.savefig("learning_curve.pdf")
    # plt.close()
    # exit(-1)

    with PdfPages(f"{name}.pdf") as pdf:
        for i, step in enumerate(log):
            theta = step["theta"]

            plt.figure(figsize=(8, 8))

            extent = X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()
            extent = 0, np.pi, 0, np.pi 

            xx = np.linspace(*extent[0:2], density)
            yy = np.linspace(*extent[2:4], density)
            xx, yy = np.meshgrid(xx, yy)
            Xfull = np.c_[xx.ravel(), yy.ravel()]

            # Xfull = np.random.rand(density**2,2)*np.pi

            # View probabilities:
            scores_full = np.array([circuit(theta, x=x) for x in Xfull])

            vmin, vmax = -1, 1

            scores = np.array([circuit(theta, x=x) for x in X])
            y_pred = sgn(scores)

            print(metrics.confusion_matrix(y, y_pred))

            accuracy = metrics.accuracy_score(y, y_pred)
            plt.title(f"Classification score, accuracy={accuracy:1.2f} ")
            plt.xlabel("feature 1")
            plt.ylabel("feature 2")

            imshow_handle = plt.contourf(xx, yy, scores_full.reshape((density, density)),  vmin=vmin, vmax=vmax, cmap='seismic')

            plt.xticks(np.linspace(0,np.pi,5))
            plt.yticks(np.linspace(0,np.pi,5))

            for cls_val, cls_col in {0:'b', 1:'r'}.items():
                # get row indexes for samples with this class
                row_ix = np.where(y == cls_val)
                # create scatter of these samples
                plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='seismic', c=cls_col, lw=1, edgecolor='k')

            ax = plt.axes([0.91, 0.1, 0.02, 0.8])
            plt.colorbar(imshow_handle, cax=ax, orientation='vertical')
            plt.clim(-1,1)

            pdf.savefig()
            plt.close()

plot(X, y, log, name="moons")
