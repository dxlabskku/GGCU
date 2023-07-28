# Gated Graph Convolutional Unit
This repository contains the Pytorch implementation code for the paper "Gated Graph Convolutional Unit"

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
You can run node classification or link prediction with the following commands.

```
python train_node_classification.py
python train_link_prediction.py
```

You can use the following commands if you want to run with GPUs.

```
python train_node_classification.py device=cuda
python train_link_prediction.py device=cuda
```

## Hyperparameters
