Setup:  
```shell
git clone https://bitbucket.org/ideas_ncbr/where-to-look-next && cd where-to-look-next 
conda env create -f environment.yml -n wtln # we recommend using mamba instead of conda (better performance)
conda activate wtln
```
Download and unzip MS COCO 14 (25 GB, reconstruction task only):
```shell
mkdir data && cd data && mkdir coco
curl -O http://images.cocodataset.org/zips/train2014.zip -O http://images.cocodataset.org/zips/val2014.zip -O http://images.cocodataset.org/zips/test2014.zip
unzip train2014.zip -d coco
unzip val2014.zip -d coco
unzip test2014.zip -d coco
rm *.zip && cd ..

```
Example:
Run AttentionMAE on MS COCO 14 with reconstruction task
```shell
python train.py Coco2014Reconstruction AttentionMae  --data-dir DATASET_DIR
```


