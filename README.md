📘 README – Running BARF on myvideo

This repository contains an implementation of BARF (Bundle-Adjusting NeRF) modified to work with your own custom dataset myvideo, stored under:

data/blender/myvideo/


The repo also supports standard Blender-style datasets with transforms_train.json, transforms_val.json, and transforms_test.json.

This guide explains how to set up, prepare your dataset, train, evaluate, and access outputs for the run named myvideo_run under the experiment group test_group.

🛠️ 1. Installation

Create a Python environment and install dependencies:

conda create -n barf-env python=3.10 -y
conda activate barf-env

pip install -r requirements.txt


OR install dependencies manually if no requirements file exists.

📁 2. Repository Structure (Important Paths)

bundle-adjusting-NeRF/
│
├── data/
│   ├── blender/
│   │   └── myvideo/
│   │       ├── images/                 # video frames
│   │       ├── transforms_train.json   # train split
│   │       ├── transforms_val.json     # validation split
│   │       └── transforms_test.json    # test split
│   │
│   ├── myvideo.py                      # custom dataset loader definition
│   ├── blender.py, llff.py, iphone.py  # other dataset loaders
│
├── model/
│   └── ...                             # BARF model files
│
├── output/
│   └── test_group/
│       ├── myvideo_run/                # main run (checkpoints, logs, renders)
│       └── myvideo_quick/
│
├── train.py
├── evaluate.py
├── extract_mesh.py
├── options.py
└── camera.py


🎞️ 3. Preparing the myvideo Dataset

Your dataset must be inside:

data/blender/myvideo/


It MUST contain:

data/blender/myvideo/
├── images/                 # extracted video frames
├── transforms_train.json
├── transforms_val.json
└── transforms_test.json


The JSON files must follow the Blender NeRF format, for example:

{
  "camera_angle_x": 0.69111122,
  "frames": [
    {
      "file_path": "images/frame00001.png",
      "transform_matrix": [
        [0.999, 0.001, 0.001, 0.0],
        [-0.001, 0.999, 0.001, 0.0],
        [-0.001, -0.001, 0.999, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    },
    ...
  ]
}


Make sure myvideo.py correctly wraps this data format for the training pipeline.

🚀 4. Training BARF on myvideo

Run the following command from the repository root:

python3 train.py \
  --group=test_group \
  --model=barf \
  --yaml=barf_blender \
  --name=myvideo_run \
  --data.dataset=myvideo \
  --data.root=data/blender/myvideo \
  --barf_c2f='[0.1,0.5]' \
  --max_iter=70000 \
  --freq.ckpt=7000 \
  --data.num_workers=0 \
  --data.preload=False


Training outputs will be saved to:

output/test_group/myvideo_run/
├── ckpts/
├── events/
├── train_renders/
└── logs.txt


Key Arguments Explained

Flag

Meaning

--group=test_group

Folder under output/

--name=myvideo_run

Subfolder name inside the group

--yaml=barf_blender

BARF config file

--data.dataset=myvideo

Tells code to use data/myvideo.py

--data.root=data/blender/myvideo

Path to your dataset

--barf_c2f='[0.1,0.5]'

Coarse-to-fine pose schedule

--max_iter=70000

Total training iterations

--freq.ckpt=7000

Save checkpoint every 7k iters

🧪 5. Evaluating the Trained Model

To run evaluation using the latest checkpoint:

python3 evaluate.py \
  --group=test_group \
  --model=barf \
  --yaml=barf_blender \
  --name=myvideo_run \
  --data.dataset=myvideo \
  --data.root=data/blender/myvideo \
  --resume


This loads:

output/test_group/myvideo_run/ckpts/latest.pth


Evaluation outputs appear under:

output/test_group/myvideo_run/eval/


Typically includes renderings, metrics (PSNR, SSIM, LPIPS), and visualizations.

🗂️ 6. Checkpoints & Large Files

Checkpoints are created every 7000 iterations:

output/test_group/myvideo_run/ckpts/iter_7000.pth
output/test_group/myvideo_run/ckpts/iter_14000.pth
...


⚠️ Important: If a checkpoint exceeds 5 GB, DO NOT push it to git. Use GitHub Releases instead, or HuggingFace Hub.

🧩 7. Troubleshooting

1. GPU Out of Memory

Reduce image resolution in transforms JSON.

Lower batch size (if configurable).

Close other GPU processes.

2. “Cannot load dataset”

Verify both flags:

--data.dataset=myvideo
--data.root=data/blender/myvideo


And ensure that myvideo.py implements the dataset class correctly.

3. Checkpoint not loading

Ensure you are using the same values for:

group

name

model

YAML file

4. Wrong image paths

Check JSON paths:

"file_path": "images/frame00001.png"


Paths should be relative to the dataset root (data/blender/myvideo).

🏁 8. Quick Start (Summary)

# Train
python3 train.py --group=test_group --model=barf --yaml=barf_blender \
  --name=myvideo_run --data.dataset=myvideo \
  --data.root=data/blender/myvideo --barf_c2f='[0.1,0.5]' \
  --max_iter=70000 --freq.ckpt=7000

# Evaluate
python3 evaluate.py --group=test_group --model=barf \
  --yaml=barf_blender --name=myvideo_run \
  --data.dataset=myvideo --data.root=data/blender/myvideo --resume
