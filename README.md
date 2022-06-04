# Reducing-Oversmoothing

This is an implementation of "Reducing Over-smoothing in Graph Neural Networks using Relational Embeddings" in Pytorch.

## Introduction

Graph Neural Networks (GNNs) have achieved a lot of success on graph-structured data. However, it is observed that the performance of GNNs does not improve (or even worsens) as the number of layers increases. This effect has known as over-smoothing, which means that the representations of the graph nodes of different classes would become indistinguishable when stacking multiple layers. In this work, we propose a new simple, and efficient method to alleviate the effect of the over-smoothing problem in GNNs by explicitly using relations between node embeddings. GNN applications are endless and depend on the user's objective and the type of data that they possess. The solving over-smoothing issue can immediately improve the performance of models on all these tasks.

## Setup
Create environment using conda

```shell
# Create environment 
conda env create -f gnn_gpu.yml
# Activate environment
conda activate gnn_gpu
# If you are working on CPU, change gnn_gpu to gnn_cpu
```

## Training and Evaluation


```shell
python main.py --log_every 100 --data cora --model DeepGAT --nlayer -1 --seed -1 --name example
python main.py --log_every 100 --data cora --model DeepGAT --nlayer -1 --seed -1 --name example2 --abs_difference
python main.py --log_every 100 --data cora --model DeepGAT --nlayer -1 --seed -1 --name example3 --abs_difference --elem_product
```

Argument explanations:
* --data: we include ['Cora', 'Pubmed', 'Citeseer']
* --nlayer: number of layers in GAT (e.g. 2). If we set it to -1, then model will train 16 GAT models with layers [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 28, 32]
* --seed: specify seed (e.g. 42). If we set set it to -1, then model will train 4 times with seeds [42, 7, 17, 37].
* --difference: use node relation - difference (h1 - h2')
* --abs_difference: use node relation - abs. difference (|h1 - h2|)
* --elem_product: use node relation - elemnt wise product (h1 * h2)

`main.py` will save checkpoints and all metrics in /results/{name}/{seed} folder for each model. We can run  `collect_results.py` to aggreagate results of all models into one document in /experiment_results.
```shell
python collect_results.py --dir "results/exp_name/cora/seed_num" --outfile_name exp-results.json
```


