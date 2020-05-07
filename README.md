## Introduction

This repository contains implementations of 5 classical zero-shot algorithms (**SAE**, **ALE**, **SJE**, **ESZSL**, and **DeViSE**) in the usual as well as the Generalized zero-shot learning (GZSL) settings using the 
`Proposed Split` and evaluation protocols (eg. Class-Averaged Top-1 Accuracy) outlined in 
[Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly](https://arxiv.org/abs/1707.00600) (**ZSLGBU**) by Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata (TPAMI 2018).

This is the **first public implementation** of `SAE`, `ALE`, `SJE` and `DeViSE` under the **ZSLGBU** protocol. An existing implementation of `ESZSL` can be found [here](https://github.com/sbharadwajj/embarrassingly-simple-zero-shot-learning) (thanks to [@sbharadwajj](https://github.com/sbharadwajj)). To this, I have added the GZSL functionality.

## Reference Papers

The original papers corresponding to the 5 algorithms are:

[1] SAE (Semantic Autoencoder) - [Semantic Autoencoder for Zero-Shot Learning](https://arxiv.org/abs/1704.08345).
Elyor Kodirov, Tao Xiang, Shaogang Gong.
CVPR, 2017.

[2] ALE (Attribute Label Embedding) - [Label-Embedding for Image Classification](https://arxiv.org/abs/1503.08677).
Zeynep Akata, Florent Perronnin, Zaid Harchaoui, Cordelia Schmid.
TPAMI, 2016.

[3] SJE (Structured Joint Embedding) - [Evaluation of Output Embeddings for Fine-Grained Image Classification](https://arxiv.org/abs/1409.8403).
Zeynep Akata, Scott Reed, Daniel Walter, Honglak Lee, Bernt Schiele.
CVPR, 2015.

[4] ESZSL - [An embarrassingly simple approach to zero-shot learning](http://proceedings.mlr.press/v37/romera-paredes15.pdf).
Bernardino Romera-Paredes, Philip H. S. Torr.
ICML, 2015.

[5] DeViSE - [DeViSE: A Deep Visual-Semantic Embedding Model](http://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf).
Andrea Frome*, Greg S. Corrado*, Jonathon Shlens*, Samy Bengio, Jeffrey Dean, Marc’Aurelio Ranzato, Tomas Mikolov.
NIPS, 2013.

## Data Splits

|Dataset |Total Images|Attributes|Class Split (Tr+Val+Ts)||ZSL      ||||GZSL           |||
|--------|------------|----------|-----------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|        |            |          |                       |tr |val|ts |tr|val|tr+val|ts seen|ts unseen|
| SUN    |14340       |102       |580+65+72              |11600|1300|1440|9280|1040|10320|2580|1440|
| CUB    |11788       |312       |100+50+50              |5875|2946|2967|4702|2355|7057|1764|2967|
| AWA1   |30475       |85        |27+13+10               |16864|7926|5685|13460|6372|19832|4958|5685|
| AWA2   |37322       |85        |27+13+10               |20218|9191|7913|16187|7340|23527|5882|7913|
| aPY    |15339       |64        |15+5+12                |6086|1329|7924|4906|1026|5932|1483|7924|

## Code

Each folder above has its own `README` with running instructions, results and their comparisons with those reported in [ZSLGBU](https://arxiv.org/abs/1707.00600). I have also put existing code references wherever relevant.

## Setup

```
git clone https://github.com/mvp18/Popular-ZSL-Algorithms.git
cd Popular-ZSL-Algorithms
bash setup.sh
```

This downloads data (splits, Res101 features and class embeddings) corresponding to the `Proposed Split` for AWA1, AWA2, CUB, SUN and aPY. To know more about the individual files, refer to the `README.txt` file available inside `xlsa17` folder.

## TODOs

- [ ] GZSL expts for ALE
- [ ] GZSL expts for DeViSE
- [ ] GZSL expts for SJE

## Contributing

If you find any errors, kindly raise an issue and I will get back to you ASAP.