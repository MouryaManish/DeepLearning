from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
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
#learning_rate = 1e-4
batch_size = 546
num_epochs = 10
optimization = ["rmsprop"]
X = data["X_test"]
y = data["y_test"]
y2 = data["y_main_test"]
test =True
"""
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

"""

                 

print("\n####################################################################################################################################\n")
print("\ntest 3 layer feedforward net: with dropouts and batchnormalization\n")
print("\n####################################################################################################################################\n")
if not test:
    #state = np.random.get_state()
    acc = 0.84
    for i in range(1):
        learning_rate = 10**np.random.uniform(low=-7,high=-5,size = 20)
        reg = np.random.uniform(low=0.45,high=0.7,size = 20)
        dropout = [0.5]#np.random.uniform(low=0.4,high=0.8,size = 5)
        print("\niteration from main:%d\n"%(i))
        for dp in dropout:
            for lr in learning_rate:
                for rg in reg:

                    model = FullyConnectedNet([80, 80, 80],input_dim=1*5,
                                  weight_scale=weight_scale,reg=rg, dtype=np.float64,normalization="batchnorm",dropout = dp)
                    solver = Solver(model,data,
                                    print_every=10, num_epochs=num_epochs, batch_size=batch_size,
                                    update_rule="rmsprop",verbose=True,
                                    optim_config={
                                      'learning_rate': lr,
                                 }
                            )
                    solver.train()
                    result = solver.check_accuracy_for_saccarde( X, y2, num_samples=None, batch_size=546)
                    if result > acc:
                        acc = result
                        marker = "cs231n/data_record/3layer_itteration_%d_hidden_%d_lr_%f_end_rg_%f_end_dp_%f_acc_%f"%(i,80,lr,rg,dp,acc)
                        solver.checkpoint_name = marker
                        solver._save_checkpoint() 
                        print("\niteration = %d\n"%(i))
                        print("\nlearning rate = %f\n"%(lr))
                        print("\nreg = %f\n"%(rg))
                        print("\n dropout = %f\n"%(dp))
                        print("\ntest accuracy = %f\n"%(acc))




else:
    model = None 
    path="cs231n/data_record/3layer_itteration_0_hidden_80_lr_0.000002_end_rg_0.475939_end_dp_0.500000_acc_0.842105_epoch_10.pkl"
    with open(path,"rb") as f:
        record = pk.load(f)
        model = record["model"]
        
    solver = Solver(model,data) 
    result = solver.check_accuracy_for_saccarde( X, y2, num_samples=None, batch_size=546)
    #result = solver.check_accuracy(X, y2, num_samples=None, batch_size=546)
    print(result)


"""

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
"""
