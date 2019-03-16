import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
#from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver
from cs231n.im2col import *
from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
from time import time
from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast
from cs231n.layer_utils import conv_relu_pool_forward, conv_relu_pool_backward
from cs231n.layer_utils import conv_relu_forward, conv_relu_backward
from data_for_project.importData import get_data
################################################################################################################################################
print("\nrelative error\n")
#################################################################################################################################################

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_data()
for k, v in list(data.items()):
    print(('%s: ' % k, v.shape))


X = data["X_test"]
y = data["y_test"]



################################################################################################################################################
print("\n3 layer cnn\n")
#################################################################################################################################################
num_epochs = 15
optimization =["rmsprop","adam","sgd"]
batch_size= 40
print("\n no. of epoch%d\n"%(num_epochs))
print("\n batch_size%d\n"%(batch_size))
#model  = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001,normalization="batchnorm",dropout=0.5)
model  = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)


for i in optimization:
    print("\nusing rule: %s\n"%(i))
    solver = Solver(model, data,
                    num_epochs=num_epochs, batch_size=40,
                    update_rule=i,
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=20)
    solver.train()

    acc = solver.check_accuracy(X, y, num_samples=None, batch_size=65)
    print("\ntest accuracy: %f\n"%(acc))


################################################################################################################################################
print("\n3 layer cnn with dropouts and batchnormalization\n")
#################################################################################################################################################

model  = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001,normalization="batchnorm",dropout=0.5)
#model  = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001,dropout=0.5)

for i in optimization:
    print("\nusing rule: %s\n"%(i))
    solver = Solver(model, data,
                    num_epochs=num_epochs, batch_size=40,
                    update_rule=i,
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=20)
    solver.train()

    acc = solver.check_accuracy(X, y, num_samples=None, batch_size=65)
    print("\ntest accuracy: %f\n"%(acc))

################################################################################################################################################
print("\n3 layer cnn with batchnormalization\n")
#################################################################################################################################################

model  = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001,normalization="batchnorm")
#model  = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001,dropout=0.5)

for i in optimization:
    print("\nusing rule: %s\n"%(i))
    solver = Solver(model, data,
                    num_epochs=num_epochs, batch_size=40,
                    update_rule=i,
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=20)
    solver.train()

    acc = solver.check_accuracy(X, y, num_samples=None, batch_size=65)
    print("\ntest accuracy: %f\n"%(acc))

################################################################################################################################################
print("\n3 layer cnn with dropouts\n")
#################################################################################################################################################

#model  = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001,normalization="batchnorm",dropout=0.5)
model  = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001,dropout=0.5)

for i in optimization:
    print("\nusing rule: %s\n"%(i))
    solver = Solver(model, data,
                    num_epochs=num_epochs, batch_size=40,
                    update_rule=i,
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    verbose=True, print_every=20)
    solver.train()

    acc = solver.check_accuracy(X, y, num_samples=None, batch_size=65)
    print("\ntest accuracy: %f\n"%(acc))

