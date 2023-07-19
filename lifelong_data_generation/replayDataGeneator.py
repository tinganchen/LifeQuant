import pandas as pd
import os
import glob
import argparse
import numpy as np
import itertools
import json

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type = str, default = 'mnist/', help = 'Data directory')
parser.add_argument('--data_csv', type = str, default = 'mnist.csv', help = 'Output csv')
parser.add_argument('--num_tasks', type = int, default = 2, help = 'Number of tasks')
parser.add_argument('--replay_rate', type = float, default = 0.2, help = 'Proportion of classes retrained in the next task')
parser.add_argument('--seed', type = int, default = 123, help = 'Number of tasks')

args = parser.parse_args()

def main(args):
    for task_id in range(args.num_tasks):
        input_csv = os.path.join(args.data_dir, args.data_csv[:-4] + f'_train{task_id}' + '.csv')
        
        train_data = pd.read_csv(input_csv)
        
        replay_data = train_data.sample(frac=args.replay_rate, replace=False, random_state=args.seed)    
        
        output_csv = input_csv.replace('train', 'replay')
        
        replay_data.to_csv(f'{output_csv}', index = False)
    
    

if __name__ == '__main__':
    main(args)