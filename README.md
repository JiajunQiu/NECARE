# NECARE
NEtwork-based CAncer gene RElationship prediciton

NECARE is a network-based algrithom which use Relational Graph Convolutional Network (R-GCN) to predict genes inetraction in cancer

The feature NECARE used include:1) knowledge-based feature OPA2Vec;                                                                                                         2)cancer specific feature which means mutation and expression profile of each gene from TCGA
                                
The ouput is binary (1 or 0) and directional.

For example, if the input genes are 'TP53 KRAS' (tab-delimited), then output '1' means TP53 has a interaction to KRAS.

To check the inetraction from KRAS to TP53, you need to set input to be 'KRAS TP53' (tab-delimited).

## How to install
NECARE is programmed bsaed in pytorch(with cuda)(python=3.7.4), so you need to install pytoch first:

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

After then, you need install:

numpy=1.16.5

dgl-cu101=0.4.1

To be eaier, you can also simply apply my conda environment by :
conda create  --name necare --file requirements.txt

or:

conda env create -f environment.yml
