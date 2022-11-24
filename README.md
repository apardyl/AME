Setup:  
```shell
git clone https://bitbucket.org/ideas_ncbr/where-to-look-next && cd where-to-look-next 
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install requirements.txt
```
Download and unzip MS COCO (25 GB, reconstruction task only):
```shell
mkdir data && cd data && mkdir coco
curl -O http://images.cocodataset.org/zips/train2014.zip -O http://images.cocodataset.org/zips/val2014.zip -O http://images.cocodataset.org/zips/test2014.zip
unzip train2014.zip -d coco
unzip val2014.zip -d coco
unzip test2014.zip -d coco
rm *.zip && cd ..
```
Run finetuning of MAE with random glimpse selection
```shell
CUDA_VISIBLE_DEVICES=0 python train.py --arch RandomMae --num-samples 50000 --epochs 10 --num-workers 8 --batch-size 64 --lr 1e-5
```

Download ADE20K:
```shell
wget -O ADEChallengeData2016.zip http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
```
Change directories to those SegmentationDataset requires