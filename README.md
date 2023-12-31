# GGCU: Reviving Gate Mechanism in GNNs with Non-Attributed Graphs
This repository contains the Pytorch implementation code for the paper "GGCU: Reviving Gate Mechanism in GNNs with Non-Attributed Graphs"

## Architectures

<img src=https://github.com/dxlabskku/GGCU/assets/117340491/d6f04ed1-2c04-426d-9a0d-59e0a566126f.jpg width="50%"/>

$\mathbf{h}^{(l-1)}$ and $\mathbf{x}$ pass through graph convolutional operation to form *global state* and *local state*. *Forget gate* and *update gate* determine the ratio between two states. $\mathbf{h}^{(l)}$ is figured out by two states and graph residual connection from $\mathbf{h}^{(l-1)}$.

*Forget gate* $f^{(l)}$ and *update gate* $u^{(l)}$ of $l$-th unit are formulated as follows:

$$f^{(l)} = 1 + \alpha \cdot \tanh(\mathbf{W}_f \cdot (\mathbf{s}^{(l)}_g \odot \mathbf{s}^{(l)}_l)).$$

$$u^{(l)} = 1 + \alpha \cdot \tanh(\mathbf{W}_u \cdot (\mathbf{s}^{(l)}_g \odot \mathbf{s}^{(l)}_l)).$$

$\mathbf{W}_f$ and $\mathbf{W}_u$ are trainable parameters, and $\odot$ is concatentation of two vectors.

## Dependencies
- CUDA 11.0
- python 3.10.9
- pytorch 1.13.1
- torch-geometric 2.2.0
- torchmetrics 0.11.1
- numpy 1.24.2
- hydra-core 1.3.2

##  Datasets
We used four benchmark datasets; Cora, CiteSeer, PubMed, and Flickr. The [data/](https://github.com/dxlabskku/GGCU/tree/main/data) folder contains the Cora benchmark dataset. You can refer to torch-geometric documentation to use other datasets [here](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html).

## Results
Testing accuracy of Node Classification without node features are summarized below.

<table>
  <tr>
    <td><b>Method</b></td>
    <td><b>Feature</b></td>
    <td><b>Cora</b></td>
    <td><b>CiteSeer</b></td>
    <td><b>PubMed</b></td>
    <td><b>Flickr</b></td>
  </tr>
  <tr>
    <td rowspan="2">GCN</td>
    <td>One-hot</td>
    <td align="right">69.96</td>
    <td align="right">45.00</td>
    <td align="right">63.04</td>
    <td align="right">51.36</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">71.06</td>
    <td align="right">46.82</td>
    <td align="right">74.40</td>
    <td align="right">54.75</td>
  </tr>
  <tr>
    <td rowspan="2">GAT</td>
    <td>One-hot</td>
    <td align="right">67.80</td>
    <td align="right">45.26</td>
    <td align="right">66.50</td>
    <td align="right">52.68</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">69.54</td>
    <td align="right">45.16</td>
    <td align="right">75.22</td>
    <td align="right">54.02</td>
  </tr>
  <tr>
    <td rowspan="2">GraphSAGE</td>
    <td>One-hot</td>
    <td align="right">61.38</td>
    <td align="right">36.12</td>
    <td align="right">55.18</td>
    <td align="right">-</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">70.96</td>
    <td align="right">45.66</td>
    <td align="right">73.52</td>
    <td align="right">-</td>
  </tr>
  <tr>
    <td rowspan="2">GGCU</td>
    <td>One-hot</td>
    <td align="right"><b>73.92</b></td>
    <td align="right"><b>52.10</b></td>
    <td align="right">72.42</td>
    <td align="right"><b>55.04</b></td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">72.98</td>
    <td align="right">46.34</td>
    <td align="right"><b>76.30</b></td>
    <td align="right">54.65</td>
  </tr>
</table>

Testing accuracy of Link Prediction without node features are summarized below.

<table>
  <tr>
    <td><b>Method</b></td>
    <td><b>Feature</b></td>
    <td><b>Cora</b></td>
    <td><b>CiteSeer</b></td>
    <td><b>PubMed</b></td>
  </tr>
  <tr>
    <td rowspan="2">GCN</td>
    <td>One-hot</td>
    <td align="right">62.29</td>
    <td align="right">62.86</td>
    <td align="right">64.17</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">76.06</td>
    <td align="right">79.49</td>
    <td align="right">79.66</td>
  </tr>
  <tr>
    <td rowspan="2">GAT</td>
    <td>One-hot</td>
    <td align="right">64.05</td>
    <td align="right">63.30</td>
    <td align="right">64.77</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">77.80</td>
    <td align="right">80.11</td>
    <td align="right">80.17</td>
  </tr>
  <tr>
    <td rowspan="2">GraphSAGE</td>
    <td>One-hot</td>
    <td align="right">67.84</td>
    <td align="right">62.24</td>
    <td align="right">64.02</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">75.59</td>
    <td align="right">78.15</td>
    <td align="right">76.58</td>
  </tr>
  <tr>
    <td rowspan="2">GGCU</td>
    <td>One-hot</td>
    <td align="right">70.02</td>
    <td align="right">68.00</td>
    <td align="right">69.73</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">77.20</td>
    <td align="right">79.49</td>
    <td align="right">77.78</td>
  </tr>
  <tr>
    <td rowspan="2">GGCU $_{trends}$</td>
    <td>One-hot</td>
    <td align="right">70.68</td>
    <td align="right">68.64</td>
    <td align="right">69.97</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right"><b>81.40</b></td>
    <td align="right"><b>81.96</b></td>
    <td align="right"><b>81.84</b></td>
  </tr>
</table>

## Usage
You can run node classification or link prediction with the one-hot representation through the following commands.

```
python train_node_classification.py
python train_link_prediction.py
```

You can use the following commands if you want to run with GPUs.

```
python train_node_classification.py device=cuda
python train_link_prediction.py device=cuda
```

You can replace the one-hot representation with the Deepwalk representation through the following commands.

```
python train_node_classification.py feature=deepwalk
python train_link_prediction.py feature=deepwalk
```

You can also run with GGCU $_{trends}$ in link prediction through the following commands.

```
python train_link_prediction.py method=trends
```

## Hyperparameters
The following hyperparameters are tuned with grid search.

Hyperparameters used in node classification with the one-hot representation of the Cora dataset, set as default values, are as follows:

<table>
  <tr>
    <td><b>Name</b></td>
    <td><b>Type</b></td>
    <td><b>Value</b></td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>float</td>
    <td>0.4</td>
  </tr>
  <tr>
    <td>n_layer</td>
    <td>int</td>
    <td>10</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>float</td>
    <td>1e-3</td>
  </tr>
  <tr>
    <td>weight_decay</td>
    <td>float</td>
    <td>5e-4</td>
  </tr>
</table>

Hyperparameters used in link prediction with the one-hot representation of the Cora dataset are as follows:

<table>
  <tr>
    <td><b>Name</b></td>
    <td><b>Type</b></td>
    <td><b>Value</b></td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>float</td>
    <td>0.1</td>
  </tr>
  <tr>
    <td>n_layer</td>
    <td>int</td>
    <td>10</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>float</td>
    <td>1e-3</td>
  </tr>
  <tr>
    <td>weight_decay</td>
    <td>float</td>
    <td>5e-4</td>
  </tr>
</table>

Hyperparameters of GGCU $_{trends}$ used in link prediction with the one-hot representation of the Cora dataset are as follows:

<table>
  <tr>
    <td><b>Name</b></td>
    <td><b>Type</b></td>
    <td><b>Value</b></td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>float</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td>n_layer</td>
    <td>int</td>
    <td>2</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>float</td>
    <td>1e-2</td>
  </tr>
  <tr>
    <td>weight_decay</td>
    <td>float</td>
    <td>5e-4</td>
  </tr>
</table>

You can change hyperparameters through the additional command "{name}={value}".

For example:

```
python train_node_classification.py alpha=0.2
```
