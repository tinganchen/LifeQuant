import pandas as pd
import os
import glob
import argparse
import numpy as np
import itertools
import json

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type = str, default = '/media/ta/e9cf3417-0c3e-4e6a-b63c-4401fabeabc8/ta/office31_split', help = 'Data directory')
parser.add_argument('--task_order', type = int, default = ['webcam', 'dslr', 'amazon'], help = 'Task order') # ['amazon', 'dslr', 'webcam']
parser.add_argument('--output_csv', type = str, default = 'office_simple.csv', help = 'Output csv')
parser.add_argument('--num_tasks', type = int, default = 3, help = 'Number of tasks')
parser.add_argument('--source', type = str, default = 'train', help = 'Train or test')
parser.add_argument('--seed', type = int, default = 123, help = 'Number of tasks')
parser.add_argument('--output_dir', type = str, default = 'office_simple/', help = 'Output directory')

args = parser.parse_args()

def main(args):
    task = args.task_order[0]
    classes = os.listdir(os.path.join(args.data_dir, task, 'images', f'{args.source}'))
    classes = np.array(sorted(classes))
    
    ll_imgs = []
    ll_labels = []
    for task in args.task_order:
        imgs = glob.glob(os.path.join(args.data_dir, task, 'images', f'{args.source}', '*/*.*'))
        imgs = [os.path.join(task, 'images', '/'.join(img.split('/')[-3:])) for img in imgs]
                                              
        # save images from classes  
        task_imgs = []
        task_labels = []
        for clss_id in range(len(classes)):
            clss = classes[clss_id]
            clss_imgs = [img for img in imgs if '/'+clss+'/' in img]
            task_imgs.extend(clss_imgs)
            task_labels.extend([clss_id]*len(clss_imgs))
            
        ll_imgs.append(task_imgs)
        ll_labels.append(task_labels)
   
    n = 4
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
  
    
    for task_id in range(args.num_tasks):
        data_csv = args.output_dir + f'{args.source}{task_id}' + args.output_csv[-n:]
        
        data_df = pd.DataFrame(columns = ['image', 'label'])
        data_df['image'] = ll_imgs[task_id]
        data_df['label'] = ll_labels[task_id]
        
        data_df.to_csv(f'{data_csv}', index = False)
    
    
    merged_imgs = list(itertools.chain(*ll_imgs))
    merged_labels = list(itertools.chain(*ll_labels))
    
    data_csv = args.output_dir + f'{args.source}' + args.output_csv[-n:]
        
    data_df = pd.DataFrame(columns = ['image', 'label'])
    data_df['image'] = merged_imgs
    data_df['label'] = merged_labels
    
    data_df.to_csv(f'{data_csv}', index = False)
    

if __name__ == '__main__':
    main(args)