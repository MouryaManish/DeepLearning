import numpy as np
import pickle as pk
path="cs231n/data_record/final_record/lr=0.000000,rg=0.755136,rep=2_epoch_5.pkl"
with open(path,"rb") as f:
    data = pk.load(f)

print(data["optim_config"]["learning_rate"])    
