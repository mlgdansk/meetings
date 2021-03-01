import pennylane as qml
from pennylane.optimize import GradientDescentOptimizer
# Create device
dev = qml.device('default.qubit', wires=1)
# Quantum node
@qml.qnode(dev)
def circuit1(var):
    qml.RX(var[0], wires=0)
    qml.RY(var[1], wires=0)
    return qml.expval(qml.PauliZ(0))
# Create optimizer
opt = GradientDescentOptimizer(0.25)
# Optimize circuit output
var = [0.5, 0.2]
for it in range(30):
    var = opt.step(circuit1, var)
    print("Step {}: cost: {}".format(it, circuit1(var)))
