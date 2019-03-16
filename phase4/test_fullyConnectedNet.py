from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
from data_for_project.importData import get_data


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

print("######################################################################################################################################")
print("DataSet:")
print("######################################################################################################################################")

data = get_data()
for k, v in list(data.items()):
  print(('%s: ' % k, v.shape))


weight_scale = 1e-2
learning_rate = 1e-4
batch_size = 40
num_epochs = 15
X = data["X_test"]
y = data["y_test"]
print("\nno. of epochs:%d\n"%(num_epochs))
print("\nbatch size:%d\n"%(batch_size))

print("\n####################################################################################################################################\n")
print("\ntest 3 layer feedforward net:\n")
print("\n####################################################################################################################################\n")

optimization = ["rmsprop","adam","sgd"]
for i in optimization:
    print("\nusing :%s\n"%(i))
    model = FullyConnectedNet([100, 100],
                  weight_scale=weight_scale, dtype=np.float64)
    solver = Solver(model,data,
                    print_every=10, num_epochs=num_epochs, batch_size=batch_size,
                    update_rule=i,
                    optim_config={
                      'learning_rate': learning_rate,
                 }
            )
    solver.train()
    acc = solver.check_accuracy(X, y, num_samples=None, batch_size=65)
    print("\ntest accuracy = %f\n"%(acc))



print("\n####################################################################################################################################\n")
print("\ntest 3 layer feedforward net: with dropouts and batchnormalization\n")
print("\n####################################################################################################################################\n")

optimization = ["rmsprop","adam","sgd"]
for i in optimization:
    print("\nusing :%s\n"%(i))
    model = FullyConnectedNet([100, 100],
                  weight_scale=weight_scale, dtype=np.float64,normalization="batchnorm",dropout = 0.5)
    solver = Solver(model,data,
                    print_every=10, num_epochs=num_epochs, batch_size=batch_size,
                    update_rule=i,
                    optim_config={
                      'learning_rate': learning_rate,
                 }
            )
    solver.train()

    acc = solver.check_accuracy(X, y, num_samples=None, batch_size=65)
    print("\ntest accuracy = %f\n"%(acc))

print("\n####################################################################################################################################\n")
print("\ntest 3 layer feedforward net: with batchnormalization\n")
print("\n####################################################################################################################################\n")

optimization = ["rmsprop","adam","sgd"]
for i in optimization:
    print("\nusing :%s\n"%(i))
    model = FullyConnectedNet([100, 100],
                  weight_scale=weight_scale, dtype=np.float64,normalization="batchnorm")
    solver = Solver(model,data,
                    print_every=10, num_epochs=num_epochs, batch_size=batch_size,
                    update_rule=i,
                    optim_config={
                      'learning_rate': learning_rate,
                 }
            )
    solver.train()

    acc = solver.check_accuracy(X, y, num_samples=None, batch_size=65)
    print("\ntest accuracy = %f\n"%(acc))

print("\n####################################################################################################################################\n")
print("\ntest 3 layer feedforward net: with dropouts\n")
print("\n####################################################################################################################################\n")

optimization = ["rmsprop","adam","sgd"]
for i in optimization:
    print("\nusing :%s\n"%(i))
    model = FullyConnectedNet([100, 100],
                  weight_scale=weight_scale, dtype=np.float64,dropout = 0.5)
    solver = Solver(model,data,
                    print_every=10, num_epochs=num_epochs, batch_size=batch_size,
                    update_rule=i,
                    optim_config={
                      'learning_rate': learning_rate,
                 }
            )
    solver.train()

    acc = solver.check_accuracy(X, y, num_samples=None, batch_size=65)
    print("\ntest accuracy = %f\n"%(acc))
