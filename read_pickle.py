import pickle
import json
from json import JSONEncoder 
import numpy as np
import torch

#read_pickle.py 
#convert reports from pickle.dump to json file into layers_json folder. 
#Need to manually choose which model the reports from. 
#File path now is for MNIST reports.

#"FashionMNIST_reports","CIFAR_reports","MNIST_reports"
file_path = "MNIST_reports"

class NdarrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def get_round_json(state_list, r_name):  
    state_dict = {}
    for clt_state in state_list:
        clt_id = clt_state[0]
        #type: collections.OrderedDict
        clt_struct = clt_state[1]
        clt_layers_dict = {}
        for layer,tensor in clt_struct.items():
            tensor_np = tensor.numpy()
            clt_layers_dict[layer] = tensor_np
        state_dict[clt_id] = clt_layers_dict

    #every round is saved in folder layers_json, with name round_state.json
    json_path = "layers_json/{}_state.json".format(r_name)
    with open(json_path,"w") as fp:
        json.dump(state_dict, fp, cls=NdarrayEncoder)

with open(file_path, "rb") as f:
    info = pickle.load(f)
    for r_name, state_list in info.items():
        if('round'in r_name):
            get_round_json(state_list, r_name)




