# Active Visual Exploration Based on Attention-Map Entropy
### [Project Page](https://io.pardyl.com/AME/) | [Paper](https://arxiv.org/abs/2303.06457)

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


