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

## How to use
Example:

For predict: NECARE.py -i ./dataset/test_pred.txt -o ./

For model training:  NECARE.py -t True -i ./dataset/test_trn.txt -g ./dataset/NECARE.graph -f ./dataset/NECARE_features.txt -s 0.1 -b 10 -e 10

Options:

  -h, --help        show this help message and exit
  
  -t TRAINING       Turn on training model of NECARE to train your own
                    modle(True/False), default is False
                    
  -i FILENAME       Iutput file (tab-delimited text file) contains the pairs
                    of input inetracitons. The first column for source genes,
                    the second column for target genes. If training model is
                    on, it need a third column for labels
                    
  -o PATH           path of the directory to save the prediction or trained
                    model, defalt is current directory
                    
  -m MODEL          The path of the modle for prediciton, default model is the
                    one we reported in NECARE paper (if using default,
                    parameters -m and -g will be ignored). -t True is
                    incompatible with -m
                    
  -g GRAPH          General gene relationship network,tab-delimited text file,
                    the first column for source genes, the second column for
                    target genes, the third column for inetraction types.
                    Default is the one used in NECARE paper
                    
  -f FEATURE        Features for the nodes, tab-delimited text file, the first
                    column for gene names, Default is the what used in NECARE
                    paper (OPA2Vec+TCGA)
                    
  -e EPOCH          Number of epoch (Only work for training model), default
                    100
                    
  -r LEARNING_RATE  Learning rate (Only work for training model), default 0.01
  
  -l HIDDEN_LAYER   Number of hidden layer (Only work for training model),
                    default 2
                    
  -n HIDDEN_NODE    Number of hidden node (Only work for training model),
                    default 100
                    
  -d DROPOUT        Rate of drapout (Only work for training model), default
                    0.2
                    
  -b BASE           Number of bases (Only work for training model), default 1
  
  -s BATCH          Batch size (Only work for training model), default 0.2
                    (20% of general gene relationship network)
