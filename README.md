# Deep Learning Hackaton

Hackaton on the obgb-ppa dataset with some added noise. To make training easier the loadData script was modified to not reload the json file every time.

The datasets were given without node features and with 7 edge features so 3 node features were engineered: Degree, Square Degree and Central Coefficent that you can see in [transform.py.](https://github.com/GiuseppeFrigeni/DL2025/blob/main/source/transforms.py) All the features were normalized. The Central Coefficent didn't give a significant boost for the accuracy and different features should have been explored. 

For the model architecture I tried: GCN, GIN, GINE, GAT, GATv2Conv and NNConv.
GINE and NNConv were the only ones that gave any sign of learning with NNConv being slightly better. In [main.py](https://github.com/GiuseppeFrigeni/DL2025/blob/main/source/main.py) you can see the hyperparameter used for NNConv are different for A,B but there was no real gain and it took more time to run.

Unfortunately doesn't seem like I am going to beat even the "Trivial Baseline" and that is mostly because i wasted too much time on try to tune a GINE architecture using a variety of different methods like Symmetric CrossEntropy, Co-teaching and Co-teaching plus, Class weigthed loss and tuning the hyperparameters.