Setup:  
```shell
git clone https://bitbucket.org/ideas_ncbr/where-to-look-next && cd where-to-look-next 
python3 -m venv venv && pip install requirements.txt
```
Download and unzip MS COCO (25 GB, reconstruction task only):
```shell
mkdir data && cd data && mkdir coco
curl -O http://images.cocodataset.org/zips/train2017.zip -O http://images.cocodataset.org/zips/val2017.zip -O http://images.cocodataset.org/zips/test2017.zip
unzip train2017.zip -d coco
unzip val2017.zip -d coco
unzip test2017.zip -d coco
rm *.zip && cd ..
```
Run U-Net for reconstruction task (no glimpses yet, input is full image)
```shell
python train.py --num-samples 100000 --num-workers 8 --batch-size 64 --lr 1e-3 --lr-decay 0.95 --weight-decay 1e-5 --arch UNet --enc-channels 3 8 16 32 64 128 --dec-channels 128 64 32 16 8
```