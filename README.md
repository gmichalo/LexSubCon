# LexSubCon: Integrating Knowledge from Lexical Resources into Contextual Embeddings for Lexical Substitution

## General info

This is the code that was used of the paper :  [LexSubCon: Integrating Knowledge from Lexical Resources into Contextual Embeddings for Lexical Substitution](https://arxiv.org/pdf/2107.05132.pdf) (ACL  2022).

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

If you find our work useful, can cite our paper using:

```
@article{Michalopoulos2021LexSubConIK,
  title={LexSubCon: Integrating Knowledge from Lexical Resources into Contextual Embeddings for Lexical Substitution},
  author={George Michalopoulos and Ian McKillop and Alexander Wong and Helen H. Chen},
  journal={ArXiv},
  year={2021},
  volume={abs/2107.05132}
}
```
