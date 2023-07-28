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
    <td>69.96</td>
    <td>45.00</td>
    <td>63.04</td>
    <td>51.36</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td>71.06</td>
    <td>46.82</td>
    <td>74.40</td>
    <td>54.75</td>
  </tr>
  <tr>
    <td rowspan="2">GAT</td>
    <td>One-hot</td>
    <td>67.80</td>
    <td>45.26</td>
    <td>66.50</td>
    <td>52.68</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td>69.54</td>
    <td>45.16</td>
    <td>75.22</td>
    <td>54.02</td>
  </tr>
  <tr>
    <td rowspan="2">GraphSAGE</td>
    <td>One-hot</td>
    <td>61.38</td>
    <td>36.12</td>
    <td>55.18</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td>70.96</td>
    <td>45.66</td>
    <td>73.52</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">GGCU</td>
    <td>One-hot</td>
    <td><b>73.92</b></td>
    <td><b>52.10</b></td>
    <td>72.42</td>
    <td><b>55.04</b></td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td>72.98</td>
    <td>46.34</td>
    <td><b>76.30</b></td>
    <td>54.65</td>
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
    <td>62.29</td>
    <td>62.86</td>
    <td>64.17</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td>76.06</td>
    <td>79.49</td>
    <td>79.66</td>
  </tr>
  <tr>
    <td rowspan="2">GAT</td>
    <td>One-hot</td>
    <td>64.05</td>
    <td>63.30</td>
    <td>64.77</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td>77.80</td>
    <td>80.11</td>
    <td>80.17</td>
  </tr>
  <tr>
    <td rowspan="2">GraphSAGE</td>
    <td>One-hot</td>
    <td>67.84</td>
    <td>62.24</td>
    <td>64.02</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td>75.59</td>
    <td>78.15</td>
    <td>76.58</td>
  </tr>
  <tr>
    <td rowspan="2">GGCU</td>
    <td>One-hot</td>
    <td>70.02</td>
    <td>68.00</td>
    <td>69.73</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td>77.20</td>
    <td>79.49</td>
    <td>77.78</td>
  </tr>
  <tr>
    <td rowspan="2">GGCU$_{trends}$</td>
    <td>One-hot</td>
    <td>70.68</td>
    <td>68.64</td>
    <td>69.97</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td><b>81.40</b></td>
    <td><b>81.96</b></td>
    <td><b>81.84</b></td>
  </tr>
</table>

## Usage
```
python train.py
```

