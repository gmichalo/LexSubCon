# LexSubCon: Integrating Knowledge from Lexical Resources into Contextual Embeddings for Lexical Substitution

## General info

This is the code that was used for the paper :  [LexSubCon: Integrating Knowledge from Lexical Resources into Contextual Embeddings for Lexical Substitution](https://arxiv.org/pdf/2107.05132.pdf) (ACL 2022).

In this work, we introduce LexSubCon, an end-to-end lexical substitution framework based on contextual embedding models that can identify highly-accurate substitute candidates. This is achieved by combining contextual information with knowledge from structured lexical resources Our approach involves:

- Proposing a novel mix-up embedding strategy to the target word's embedding through linearly  interpolating the pair of the target input embedding and the average embedding of its probable synonym
- Considering the similarity of the sentence-definition embeddings of the target word and its proposed candidates
- Calculating the effect of each substitution on the semantics of the sentence through a fine-tuned sentence similarity model

<p align="center">
 <img src="/images/LexSubCon.png" height="250" width="500">
 </p>



## Technologies
This project was created with python 3.7 and PyTorch 1.7.1 and it is based on the transformer github repo of the huggingface [team](https://huggingface.co/)
and the github repo of the sentence-transformers [team](https://github.com/UKPLab/sentence-transformers)
## Setup
We recommend installing and running the code from within a virtual environment.

### Creating a Conda Virtual Environment
First, download Anaconda  from this [link](https://www.anaconda.com/distribution/)

Second, create a conda environment with python 3.7.
```
$ conda create -n lexsubcon python=3.7
```
Upon  restarting your terminal session, you can activate the conda environment:
```
$ conda activate lexsubcon 
```
### Install the required python packages
In the project root directory, run the following to install the required packages.
```
pip3 install -r requirements.txt
```

- The LexSubCon  model depends on data from NLTK  so you'll have to download them. Run the Python interpreter (python3) and type the commands:
```python
>>> import nltk
>>> nltk.download('punkt')
```

#### Install from a VM
If you start a VM, please run the following command sequentially before install the required python packages.
The following code example is for a vast.ai Virtual Machine.

```
apt-get update
apt install git-all
apt install python3-pip
```

## Dowload pre-trained  Gloss and Similarity checkpoint  model

In order to use pre-trained  Gloss and Similarity model, you need to dowload them  using the scripts **checkpoint/gloss_bert/download.sh** and **checkpoint/similarity_new_bert/download_4.sh** respectively.

## Datasets

This repository contains preprocessed data files from the github [repo](https://github.com/orenmel/lexsub) of the paper:

**A Simple Word Embedding Model for Lexical Substitution**
Oren Melamud, Omer Levy, Ido Dagan.  Workshop on Vector Space Modeling for NLP (VSM), 2015 [[pdf]](http://u.cs.biu.ac.il/~melamuo/publications/melamud_vsm15.pdf).

based on the datasets introduced by the following papers:

**Semeval-2007 task 10: English lexical substitution task**
Diana McCarthy, Roberto Navigli, SemEval 2007.  (dataset/LS07)

**What substitutes tell us-analysis of an ”all-words” lexical substitution corpus.**
Gerhard Kremer,Katrin Erk, Sebastian Pado,  Stefan Thater. EACL, 2014.   (dataset/LS14)


## Running model
In order to run the model for the LS07 dataset:

```
python3 main_lexical.py
```

and in order to run the model for the LS14 dataset:

```
python3 main_lexical_coinco.py
```

in order to use the difference signals you will need to set the respective flag to True:
```
    parser.add_argument("-val", "--validation_score", type=bool, help="whether we use validation score")
    parser.add_argument("-pro", "--proposed_score", type=bool, help="whether we use proposed score")
    parser.add_argument("-glossc", "--gloss_score", type=bool, help="whether we use gloss score")
    parser.add_argument("-similn", "--similarity_sentence_new", type=bool,help="whether we use similarity sentence score new")
```

### Candidate Ranking Task
Finally for testing the model on the Candidate Ranking Task, you will need to set to True the following parameter:
```
parser.add_argument("-g", "--gap", type=bool, help="whether we use the gap ranking (candidate ranking)")
```


### Running Individual Features and Combining them
Finally, in the case that a computer cannot simultaneously run  all the signals, you can run each signal individually (with weight 1) and run the  **lexical_combined_results.py** script after updating the paths of the results

### Citation
If you find our work useful, can cite our paper using:



```
@inproceedings{michalopoulos-etal-2022-lexsubcon,
    title = "{L}ex{S}ub{C}on: Integrating Knowledge from Lexical Resources into Contextual Embeddings for Lexical Substitution",
    author = "Michalopoulos, George  and
      McKillop, Ian  and
      Wong, Alexander  and
      Chen, Helen",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.87",
    pages = "1226--1236",
}
```
