# Overcoming Forgetting Catastrophe in Quantization-Aware Training (LifeQuant)
ICCV 2023 Accepted Paper - Quantization, Efficient Inference, Lifelong Learning 

<img src="img/overview.png" width="350" height="300">

* **Motivations**:

- 1. to overcome quantization models forgetting the learned space on old task data 
- 2. to employ as few as replay data (old task data) for memory efficiency, but to avoid forgetting the old tasks

* **Methods**:

- 1. **ProxQ**: to regularize the quantization space when learning new task data
- 2. **BaLL**: to strengthen the weighting of the few replay data (i.e., rebalancing)

## Requirements

* python3
* pytorch==1.7.1
* cudatoolkit==11.0.221 
* numpy==1.19.2
* tensorboardx==1.4

## Implementation

* e.g. 4-bit ResNet-20 on CIFAR-100-LL (gamma = 25).

### Data Preparation

* e.g. CIFAR-100.

```shell
cd src/
mkdir data_dir/
```

Save training (testing) image data to the path src/data_dir/cifar100/. (e.g. src/data_dir/cifar100/train/apple/0001.png)

### Data Generation - Lifelong Data

##### 1. Generate lifelong data

* e.g. CIFAR-100-LL (3 tasks with 25% classes forgot whenerver task switches).

```shell
cd lifelong_data_generation/
python3 lifeLongDataGeneator.py --data_dir "src/data_dir/cifar100/" --output_csv "cifar100.csv" --num_tasks 3 --diminish_rate 25 --source train
```

Files 'cifar100_train.csv', 'cifar100_train0.csv', 'cifar100_train1.csv', and 'cifar100_train2.csv' are shown under lifelong_data_generation/cifar100_025.

* 'cifar100_train.csv': all training data
* 'cifar100_train$i$.csv': the i-th task of training data

Testing data files can be generated by setting the argument *--source test*.

##### 2. Generate replay data

* e.g. CIFAR-100-LL (20% old task data are added to the new tasks).

```shell
python3 replayDataGeneator.py --data_dir "lifelong_data_generation/data_dir/cifar100_025/" --output_csv "cifar100.csv" --num_tasks 3 --replay_rate 0.2
```

Files 'cifar100_replay0.csv', 'cifar100_replay1.csv', and 'cifar100_replay2.csv' are shown under lifelong_data_generation/cifar100_025 as well.

* 'cifar100_replay$i$.csv': the replay data from the j-th task (for all j < i) to be added to the i-th task.

**ps.** No testing data should be replayed.

##### 3. Count number of samples of each class

This step is for our second method, **BaLL**.

```shell
cp count_classes.py lifelong_data_generation/data_dir/cifar100_025/
python3 count_classes.py
```
**ps.** Note that if the number of tasks is not 3, Lines 8-11 in count_classes.py should be modified to include all generated training data.


### Training & Testing

* e.g. 4-bit ResNet-20 on CIFAR-100-LL.

##### 1. Copy the generated data for training

```shell
cd src/resnet-20-cifar-100/
cp -r ../../lifelong_data_generation/data_dir/cifar100_025/ data/
```
Generated csv files are copied under data/.

##### 2. Pretraining

* 32-bit

```shell
python3 main.py --gpus 0 --csv_dir data/cifar100_025/ --method lifeq --lr 0.001 --num_tasks 1 --job_dir pretrained/resnet/t_32_025_0  --pretrained False --bitW 32 --abitW 32
```

* 8-bit

```shell
python3 main.py --gpus 0 --csv_dir data/cifar100_025/ --method life_q --ll_method ours  --balanced True --lr 0.001 --num_tasks 1 --job_dir experiment/life_q/ours_/resnet/t_8_32_025_0 --source_dir pretrained/ --source_file resnet/t_32_025_0/checkpoint/model_best.pt --bitW 8 --abitW 8
```

##### 2. Quantize & Test

* 4-bit

```shell
python3 main.py --gpus 0 --csv_dir data/cifar100_025/ --method life_q --ll_method ours  --balanced True --num_tasks 3 --job_dir experiment/life_q/ours_/resnet/t_4_8_025_3 --source_dir experiment/ --source_file life_q/ours/resnet/t_8_32_025_0/checkpoint/model_best.pt --bitW 4 --abitW 4
```


## Citation

```shell
@InProceedings{Chen_2023_ICCV,
    author    = {Chen, Ting-An and Yang, De-Nian and Chen, Ming-Syan},
    title     = {Overcoming Forgetting Catastrophe in Quantization-Aware Training},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {17358-17367}
}
```
