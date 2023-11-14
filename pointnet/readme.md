# PointNet

## Environment Setup

```
conda create -n week1 python=3.9
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install tqdm h5py matplotlib ninja chamferdist
```

## Classification

```
python train_cls.py [--gpu 0] [--epochs 100] [--batch_size 32] [--lr 1e-3] [--save]
```

## Part Segmentation

```
python train_seg.py [--gpu 0] [--epochs 100] [--batch_size 32] [--lr 1e-3] [--save]
```

## Auto-Encoder

```
python train_ae.py [--gpu 0] [--epochs 100] [--batch_size 32] [--lr 1e-3] [--save]
``````
