# Active Visual Exploration Based on Attention-Map Entropy (IJCAI 2023)
### [Project Page](https://io.pardyl.com/AME/) | [Paper](https://arxiv.org/abs/2303.06457)
Official PyTorch implementation of the paper: "Active Visual Exploration Based on Attention-Map Entropy"

![](https://io.pardyl.com/AME/static/images/architecture.png)



> **Active Visual Exploration Based on Attention-Map Entropy**<br>
> Adam Pardyl, Grzegorz Rypeść, Grzegorz Kurzejamski, Bartosz Zieliński, Tomasz Trzciński<br>
> [https://arxiv.org/pdf/2007.01289](https://arxiv.org/abs/2303.06457) <br>
>
>**Abstract:** Active visual exploration addresses the issue of limited sensor capabilities in real-world scenarios, where successive observations are actively chosen based on the environment. To tackle this problem, we introduce a new technique called Attention-Map Entropy (AME). It leverages the internal uncertainty of the transformer-based model to determine the most informative observations. In contrast to existing solutions, it does not require additional loss components, which simplifies the training. Through experiments, which also mimic retina-like sensors, we show that such simplified training significantly improves the performance of reconstruction, segmentation and classification on publicly available datasets.
___

## Setup
```shell
git clone https://bitbucket.org/ideas_ncbr/where-to-look-next && cd where-to-look-next 
conda env create -f environment.yml -n wtln # we recommend using mamba instead of conda (better performance)
conda activate wtln
```
## Train
* download and extract the requested dataset
* run training with:
```shell
python train.py <dataset> <model> [params]
```
where dataset is one of:
* Reconstruction task:
  * ADE20KReconstruction
  * Coco2014Reconstruction
  * Sun360Reconstruction
  * TestImageDirReconstruction
* Segmentation task:
  * ADE20KSegmentation
* Classification task:
  * Sun360Classification (for train all configuration)
  * EmbedClassification (for head-only configuration, prepare embeddings with predict.py first)

and the model is:
* {Attention/Random/Checkerboard}Mae for reconstruction
* {Attention/Random/Checkerboard}SegMae for segmentation
* {Attention/Random/Checkerboard}ClsMae for train-all classification
* EmbedClassifier for head-only classification

Example:
Run AttentionMAE on MS COCO 14 with reconstruction task
```shell
python train.py Coco2014Reconstruction AttentionMae  --data-dir DATASET_DIR
```

Run `python train.py <dataset> <model> --help` for available training params.

Visualizations form the paper can be generated using predict.py (use `--visualization-path` param).

Embeddings for head-only classification are generated with predict.py (`--dump-path`).

Average glimpse selection maps and evaluation of a trained model can be obtained with predict.py
with `--avg-glimpse-path` and `--test` params accordingly.

## Citation
If you find this useful for your research, please use the following.

```
@inproceedings{pardyl2023active,
  title={Active Visual Exploration Based on Attention-Map Entropy},
  author={Pardyl, Adam and Rype{\'s}{\'c}, Grzegorz and Kurzejamski,
            Grzegorz and Zieli{\'n}ski, Bartosz and Trzci{\'n}ski, Tomasz},
  booktitle={Proceedings of the 32nd International Joint Conference on Artificial Intelligence},
  year={2023}
}
```
