# Rethinking Efficiency and Redundancy in Training Large-scale Graphs
An implementation of DropReef: a novel method to drop the redundancy in large-scale graphs once and for all, helping improve the efficiency of training large-scale graphs with GNNs. Please follow the guideline given in [Usage](#Usage) to verify the performance of DropReef. Since DropReef is a once-for-all method performed offline, we have processed all large-scale graphs via DropReef to generate the train_adj (adjacency matrix for training) files. Therefore, one can directly load the processed adj together with other data of one dataset. A flexibly adjustable script of DropReef will be released after acceptance.


## Overview
DropReef is a novel method to detect and drop redundant nodes in large-scale graphs once and for all, promoting the efficiency of GNN training while ensuring no sacrifice in the model accuracy. DropReef consists of three subprocesses: 1) computing metrics, 2) detecting redundancy, and 3) dropping nodes. The overall procedure of DropReef is offline performed before training large-scale graphs with GNNs. The generated low-redundancy graphs are yielded by removing redundant nodes and associated edges from the original graphs. One can perform training on these low-redundancy graphs to witness a considerable acceleration straightforwardly.


## Datasets
All datasets used in our papers are available for download:
* reddit
* amazon
* yelp
* ogbn-product 
  
One can download the processed data (by DropReef) together with original datasets from [Google Drive](https://drive.google.com/drive/folders/1UM7WgCLvMX1ToMXcKn0lKG7DNkSwfNZE?usp=sharing).  
**NOTE**: Datasets used in PGS-GNN are the same as those used in GraphSAINT. Just add `adj_train_DR.npz` or `adj_train_DR_pg.npz` provided in ```PGS-GNN/data/dataset_name``` to the corresponding directory. 

The directory structure should be as below:
```
Anonymous-DropReef/
│   README.md
│   
└───GraphSAINT/ 
└─────data/
└───────Amazon/
└─────────adj_full.npz
└─────────adj_train.npz
└─────────adj_train_DR.npz
└─────────class_map.json
└─────────feats.npy
└─────────role.json
└───────Reddit/
......
|   
└───Cluster-GCN/
└─────data/
└───────Amazon/
└─────────amazon-G.json
└─────────amazon-G_DR.json
└─────────amazon-id_map.json
└─────────amazon-id_map_DR.json
└─────────amazon-feats.npy
└─────────amazon-class_map.json
└───────Reddit/
......
└───PGS-GNN/
└─────data/
└───────Amazon/
└─────────adj_full.npz
└─────────adj_train.npz
└─────────adj_train_DR_pg.npz
└─────────class_map.json
└─────────feats.npy
└─────────role.json
└───────Reddit/
......
```

## Usage
To reproduce the results of our experiments, simply go into each model directory and run the prepared scripts. Please download the datasets and place them in correct locations as given in [Dataset](#Datasets).

Run the original version (baseline):
```
./<model_name>/run_<dataset_name>.sh
```
Run the version with DropReef utilized:
```
modify the run_<dataset_name>.sh by adding `--use_DropReef` mark, and then run:
./<model_name>/run_<dataset_name>.sh
```

## Experimental Devices
| Platform | Configuration |
|---|---
| CPU | Intel Xeon CPU E5-2650 v4 CPUs (dual 24-core) |
| GPU | NVIDIA Tesla V100 GPU (16 GB memory) |


## Dependencies
* python
* pytorch
* tensorflow
* numpy
* scipy
* scikit-learn
* pyyaml
* [metis](https://github.com/google-research/google-research/tree/master/cluster_gcn) = 5.1.0
* networkx = 1.11


## Acknowledgements
The proposed DropReef is applied to recent state-of-the-art sampling-based GNNs to demonstrate the effectiveness of DropReef in terms of efficiency and accuracy. We use the implementations of [Cluster-GCN](https://github.com/google-research/google-research/tree/master/cluster_gcn), [PGS-GNN](https://github.com/ZimpleX/gcn-ipdps19), and [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT) as backbones, and owe many thanks to the authors for making their code available. Moreover, we thank the authors of [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT) for offering download links to many datasets. 
