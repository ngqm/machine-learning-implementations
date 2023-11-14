# NERF

## Environment Setup

```
conda create --name nerf python=3.8

conda activate nerf
pip install -r requirements.txt

pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torchmetrics[image]
pip install tensorboard

export PYTHONPATH=.
```

## Data

```
sh scripts/data/download_example_data.sh
```

## Qualitative Evaluation

```
python torch_nerf/runners/render.py +log_dir=${LOG_DIR} +render_test_views=False
python scripts/utils/create_video.py --img_dir ${RENDERED_IMG_DIR} --vid_title ${VIDEO_TITLE}
```

## Quantitative Evaluation

```
python torch_nerf/runners/render.py +log_dir=${LOG_DIR} +render_test_views=True
python torch_nerf/runners/evaluate.py ${RENDERED_IMG_DIR} ./data/nerf_synthetic/lego/test
```