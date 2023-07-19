import os
import pandas as pd
import numpy as np
import torch
import json


FILES = [[pd.read_csv('train0.csv')],
         [pd.read_csv('train1.csv'), pd.read_csv('replay0.csv')],
         [pd.read_csv('train2.csv'), pd.read_csv('replay0.csv'), 
          pd.read_csv('replay1.csv')]]

for task_id, FILE in enumerate(FILES):
    # count nclass
    sample_num_per_cls = dict()
    
    for f in FILE:
        for class_ in f['label']:
            if class_ in sample_num_per_cls:
                sample_num_per_cls[class_] += 1
            else:
                sample_num_per_cls[class_] = 1
    
   
    class_num = len(sample_num_per_cls)
    
    s_num_per_cls = torch.Tensor(list(sample_num_per_cls.values()))
    
    # reweighting
    '''
    beta = 0.999
    
    effective_num = 1.0 - beta**s_num_per_cls
    weights = (1.0 - beta) / effective_num
    weights = weights / torch.sum(s_num_per_cls) * class_num
    '''
    weights = 1 / ((s_num_per_cls / torch.sum(s_num_per_cls) + 1/class_num) * class_num) 
    
    reweight = dict()
    
    for i, cls in enumerate(sample_num_per_cls.keys()):
        reweight[cls] = weights[i].item()
    
    with open(f"reweight_{task_id}.json", "w") as outfile:
        json.dump(reweight, outfile)
    
    print(reweight)

#with open('reweight_0.json') as json_file:
#    data = json.load(json_file)