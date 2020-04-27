#calculate the difference use cosine similarity
import pickle
import json
from json import JSONEncoder 
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
import math
file_path = "MNIST_reports"

#Set the threshold for similarity
EPI = 0

def euclidean_distance(arr1, arr2):
    dim1_len = len(arr1)
    dim2_len = len(arr1[0])
    dist_list = []
    for i in range(dim1_len):
        dist_sum = 0
        for j in range(dim2_len):
            dist_sum = dist_sum + pow((arr1[i][j]-arr2[i][j]),2) 
        dist = math.sqrt(dist_sum)      
        dist_list.append(dist)
    return np.asarray(dist_list)
            

def calculate_conv_sim(ftensor, ltensor):
    #conv1: [20,1,5,5], conv2 [50,20,5,5]
    farr = ftensor.numpy()
    larr = ltensor.numpy()
    dim1_len = len(farr)
    dim2_len = len(farr[0])
    sim_list = []
    #loop through the outest dim
    for i in range(dim1_len):
        #loop through second outest dim
        dim2_sim = []
        for j in range(dim2_len):
            inner_sim = euclidean_distance(farr[i][j],larr[i][j])
            dim2_sim.append(inner_sim)
        sim_list.append(dim2_sim)
    sim_ndarr = np.asarray(sim_list)
    return sim_ndarr 

def calculate_fc_sim(ftensor, ltensor):
    #fc1 [500,800] fc2[10,500]
    #use 
    farr = ftensor.numpy()
    larr = ltensor.numpy()
    sim_ndarr = euclidean_distance(farr,larr)
    return sim_ndarr


#sim_dict {'1':{'layers': similarvalue},'2':{'layers': similarvalue}}
def get_sim_dict(client_info): 
    #sim_dict {'1':{'layers': similarvalue},'2':{'layers': similarvalue}} 
    sim_dict = {}
    #{id: {round: weights, round: weights}, id:{}, ...}
    for clt_id, r_list in client_info.items():
        if(len(r_list) == 1):
            continue
        
        #flayers: {layername: weights_ndarray}
        flayers = r_list[0][1]
        llayers = r_list[-1][1]

        sim_layers_dict = {}
        for layername in flayers.keys():
            #only calculate weight
            if('weight' in layername):
                if('conv' in layername):
                    sim_ndarr = calculate_conv_sim(flayers[layername], llayers[layername])
                if('fc' in layername):
                    sim_ndarr = calculate_fc_sim(flayers[layername], llayers[layername])

                sim_layers_dict[layername] = sim_ndarr

        sim_dict[clt_id] = sim_layers_dict
        
    return sim_dict


def get_update_layers_list(sim_dict):
    #sim_dict {'1':{'layers': similarvalue},'2':{'layers': similarvalue}} 
    #get update layers lists wrt clients
    update_layers = {}

    for client_id, clt_dict in sim_dict.items():
        clt_update = []
        for clt_layername, clt_sim in clt_dict.items():
            if('weight' in clt_layername):   
                print(clt_layername, clt_sim.shape)
                #clt_sim is a matrix! how to set threshold? 
                #if(clt_sim.any()):
                clt_update.append(clt_layername)
        update_layers[client_id] = clt_update

    return update_layers


with open(file_path, "rb") as f:
    info = pickle.load(f)

    #{id: {round: weights, round: weights}, id:{}, ...}
    client_info = {}
    clt_id_list = []
    for r_name, v_list in info.items():
        if('round'in r_name):
            clt_keys = list(client_info.keys())
            for v_tuple in v_list:
                item = {}
                clt_id = v_tuple[0]
                clt_id_list.append(clt_id)
                if(clt_id not in clt_keys):
                    client_info[clt_id] = []
                client_info[clt_id].append([r_name, v_tuple[1]])              
    sim_dict = get_sim_dict(client_info)
    update_layers = get_update_layers_list(sim_dict)

    #for every client, which layer need to update. 
    #print(update_layers)
    


