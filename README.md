# Deep Learning Hackaton

Hackaton on the obgb-ppa dataset with some added noise. To make training easier the loadData script was modified to not reload the json file every time.

The datasets were given without node features and with 7 edge features so 11 node features were engineered: Degree, Square Degree,Central Coefficent and 8 Laplacian eigenvector positional encoding that you can see in [transform.py.](https://github.com/GiuseppeFrigeni/DL2025/blob/main/source/transforms.py) All the features were normalized.

For the model architecture I tried: GCN, GIN, GINE, GAT, GATv2Conv, GNN+(state of the art on obgb-ppa) and NNConv.
GINE and NNConv were the only ones that gave any sign of learning.

I tried a  variety of different methods like Symmetric CrossEntropy, Co-teaching and Co-teaching plus, but the thing that worked best in the end was recreating the GCOD loss from the paper [Wani et al. (2024).](https://arxiv.org/abs/2412.08419)

 In [main.py](https://github.com/GiuseppeFrigeni/DL2025/blob/main/source/main.py) you can see the hyperparameter used are the same of the paper linked. Unfortunately I finished all the GPU time even after buying Colab Pro on both Kaggle and Colab and I couldn't train the model for more epochs and try more configurations of hyperparameters for the GCOD approach.
