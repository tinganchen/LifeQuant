import pandas as pd
import os
import glob
import argparse
import numpy as np
import itertools
import json

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type = str, default = '/media/ta/e9cf3417-0c3e-4e6a-b63c-4401fabeabc8/ta/mnist', help = 'Data directory')
parser.add_argument('--output_csv', type = str, default = 'mnist.csv', help = 'Output csv')
parser.add_argument('--num_tasks', type = int, default = 2, help = 'Number of tasks')
parser.add_argument('--diminish_rate', type = float, default = 1, help = 'Proportion of classes remain to the the next task')
parser.add_argument('--source', type = str, default = 'test', help = 'Train or test')
parser.add_argument('--seed', type = int, default = 123, help = 'Number of tasks')
parser.add_argument('--output_dir', type = str, default = 'mnist/', help = 'Output directory')

args = parser.parse_args()

def main(args):
    classes = os.listdir(os.path.join(args.data_dir, f'{args.source}'))
    classes = np.array(sorted(classes))
    imgs = glob.glob(os.path.join(args.data_dir, f'{args.source}', '*/*.*'))
    imgs = ['/'.join(img.split('/')[-3:]) for img in imgs]
                                          
    # sample images from classes
    np.random.seed(args.seed)
    classes_order = np.arange(len(classes))
    np.random.shuffle(classes_order)
    
    n_classes = len(classes)
    
    n_classes_task = int(n_classes / (1 + (args.diminish_rate)*(args.num_tasks-1)))
    
    task2classes = dict()
    ll_imgs = []
    ll_labels = []
    head = 0 
    for i in range(args.num_tasks):
        if i == args.num_tasks-1:
            sampled_classes = classes_order[head:]  
        else:
            sampled_classes = classes_order[head:head+n_classes_task] #classes_order[head:head+n_classes_task+1] 
        #print(sampled_classes)
        head = head+int(n_classes_task * (args.diminish_rate))
       
        task_imgs = []
        task_labels = []
        for clss_id in sampled_classes:
            clss = classes[clss_id]
            clss_imgs = [img for img in imgs if '/'+clss+'/' in img]
            task_imgs.extend(clss_imgs)
            task_labels.extend([clss_id]*len(clss_imgs))
            
        ll_imgs.append(task_imgs)
        ll_labels.append(task_labels)
        task2classes[i] = ','.join([str(l) for l in sampled_classes])
    
    n = 4
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    '''
    filename = args.output_dir + args.output_dir[:-1] + '.txt'
    
    with open(filename, 'w') as f:
        json.dump(task2classes, f)
    
    
    with open(filename, 'r') as f:
        data = json.load(f)
    '''
    
    #data
    

    
    for task_id in range(args.num_tasks):
        data_csv = args.output_dir + args.output_csv[:-n] + f'_{args.source}{task_id}' + args.output_csv[-n:]
        
        data_df = pd.DataFrame(columns = ['image', 'label'])
        data_df['image'] = ll_imgs[task_id]
        data_df['label'] = ll_labels[task_id]
        
        data_df.to_csv(f'{data_csv}', index = False)
    
    
    merged_imgs = list(itertools.chain(*ll_imgs))
    merged_labels = list(itertools.chain(*ll_labels))
    
    data_csv = args.output_dir + args.output_csv[:-n] + f'_{args.source}' + args.output_csv[-n:]
        
    data_df = pd.DataFrame(columns = ['image', 'label'])
    data_df['image'] = merged_imgs
    data_df['label'] = merged_labels
    
    data_df.to_csv(f'{data_csv}', index = False)
    

if __name__ == '__main__':
    main(args)