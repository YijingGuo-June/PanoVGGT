<div align="center">
<h1>PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery</h1>

<a href="https://arxiv.org/abs/2603.17571"><img src="https://img.shields.io/badge/arXiv-2603.17571-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/datasets/YijingGuo/PanoCity"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PanoCity_Dataset-blue' alt="Dataset"></a>
<a href="https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_Weights-orange' alt="Weights"></a>

Yijing Guo, Mengjun Chao, Luo Wang, Tianyang Zhao, Haizhao Dai, Yingliang Zhang, Jingyi Yu, Yujiao Shi
</div>

```bibtex
@article{guo2026panovggt,
  title={PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery},
  author={Guo, Yijing and Chao, Mengjun and Wang, Luo and Zhao, Tianyang and Dai, Haizhao and Zhang, Yingliang and Yu, Jingyi and Shi, Yujiao},
  journal={arXiv preprint arXiv:2603.17571},
  year={2026}
}
```

## Updates

  - [April 2026] We have officially released the high-resolution pre-trained model weights\! You can download them directly from [Hugging Face](https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt).
  - [April 2026] We have also released the low-resolution checkpoint used for the paper's quantitative evaluation. Please see the checkpoint note below before reproducing results.
  - [March 2026] The PanoVGGT paper is now available on [arXiv](https://arxiv.org/abs/2603.17571).
  - [March 2026] We have released the [PanoCity Dataset](https://huggingface.co/datasets/YijingGuo/PanoCity) on Hugging Face\!

## Overview

**PanoVGGT** is a feed-forward, geometry-aware Transformer framework designed for globally consistent 3D reconstruction directly **from unordered equirectangular images**. It serves as a complete, end-to-end panoramic 3D reconstruction pipeline, providing practical tooling for outputting depth maps, camera poses, and point clouds.

## Quick Start

First, clone this repository to your local machine and install the dependencies. We recommend using Python 3.10+ and a CUDA-enabled PyTorch environment for optimal GPU inference and training.

```bash
git clone https://github.com/YijingGuo-June/PanoVGGT.git
cd PanoVGGT

# 2. Create and activate a virtual environment (Recommended)
conda create -n panovggt python=3.11 -y
conda activate panovggt

# 3. Install PyTorch and related packages
# Note: The command below is for CUDA 12.4. You can adjust the versions 
# based on your local CUDA environment if needed.
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124

# 4. Install the remaining dependencies
pip install -r requirements.txt
```

### Download Pre-trained Weights

You can download the pre-trained weights manually or via the command line:

```bash
# Create a checkpoints directory
mkdir -p checkpoints

# Download the model weights
wget https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt -O checkpoints/model.pt
```

### Checkpoint Note

The released checkpoint [`model.pt`](https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt) is our recommended high-resolution pre-trained weight. It was released after we completed the second-stage high-resolution training and is intended as the default checkpoint for inference, fine-tuning, and downstream pre-training. Compared with the low-resolution checkpoint, it brings a substantial performance improvement under high-resolution evaluation settings, since panoramic images contain dense visual information and downsampling removes many important details.

For clarity, the quantitative results reported in our paper were **not** evaluated with `model.pt`. Similar to many VGGT-style methods, PanoVGGT follows a two-stage training recipe: first-stage training at `336 x 672`, followed by second-stage training at `518 x 1036`. During the paper deadline, the second-stage training had not yet been completed, so the paper reports results using the first-stage low-resolution checkpoint, evaluated under the high-resolution setting. In contrast, the compared methods were evaluated with their original settings, typically using high-resolution training and high-resolution testing, which makes the comparison somewhat unfavorable to PanoVGGT.

If you want to reproduce the quantitative results in the paper, please use the low-resolution checkpoint:

```bash
wget https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model_lowres.pt -O checkpoints/model_lowres.pt
```

We do **not** recommend using `model_lowres.pt` as a general pre-trained checkpoint. For most use cases, please use the high-resolution [`model.pt`](https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt).

## Repository Layout

<details>
<summary>Click to expand the project structure</summary>

```text
.
|-- app.py                 # Gradio demo entry
|-- panovggt/              # Core model, layers, geometry, projection
|-- training/              # Training configs, datasets, trainer, utils
|-- evaluation/            # Evaluation scripts and metrics
|-- examples/              # Example panoramic sequences
`-- requirements.txt       # Python dependencies
```

</details>

## Interactive Demo

We provide a Gradio web interface to run reconstructions and interactively visualize your panoramic scenes.

### Gradio Web Interface

Once you have downloaded the pre-trained weights, run the following command to launch the application locally:

```bash
python app.py \
  --config training/config/default.yaml \
  --checkpoint checkpoints/model.pt \
  --device cuda \
  --port 7860
```

**Optional arguments:**

  - `--share`: Enable a public Gradio share link to access the demo remotely.
  - `--tmp-dir`: Set a custom temporary session directory.

## Training

To train PanoVGGT from scratch or fine-tune it on your own panoramic data, you can use our distributed training launch script.

```bash
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 torchrun --standalone --nproc_per_node=8 training/launch.py
```

The main configurations are handled via YAML files located in the `training/config` directory:

  - `training/config/default.yaml` (Model & Training hyper-parameters)
  - `training/config/default_dataset.yaml` (Dataset paths & loaders)

## Evaluation

To reproduce our quantitative results and evaluate the model's geometry and pose estimations across panoramic datasets:

```bash
python evaluation/eval_allpano.py
```

### High-resolution Results

The following results are obtained with the released high-resolution checkpoint [`model.pt`](https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt), trained at `518 x 1036` and evaluated at `518 x 1036` in the multi-view fusion setting with 3 input frames per sequence.

| Dataset | # Seq. | AUC@30 ↑ | AUC@15 ↑ | AUC@5 ↑ | AUC@3 ↑ | Abs Rel ↓ | RMSE ↓ | δ < 1.25 ↑ | Global Overall Mean ↓ | Global Overall Med. ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AerialGo | 253 | 0.998 | 0.996 | 0.987 | 0.978 | 0.026 | 5.765 | 0.977 | 1.817 | 0.609 |
| Matterport3D | 129 | 0.902 | 0.854 | 0.701 | 0.581 | 0.052 | 0.218 | 0.971 | 0.059 | 0.034 |
| Stanford2D3DS | 37 | 0.917 | 0.881 | 0.809 | 0.757 | 0.046 | 0.248 | 0.968 | 0.057 | 0.033 |

## Checklist / TODOs

  - [x] Release the PanoCity Dataset
  - [x] Release the training code and evaluation scripts
  - [x] Release the Gradio demo application
  - [x] Release the pre-trained model weights
  - [ ] Release the remaining weather datasets
  - [ ] Open-source the panoramic data collection pipeline

## Citation

If you find our paper, the PanoCity dataset, or our data collection pipeline useful for your research or projects, please consider citing our work:

```bibtex
@article{guo2026panovggt,
  title={PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery},
  author={Guo, Yijing and Chao, Mengjun and Wang, Luo and Zhao, Tianyang and Dai, Haizhao and Zhang, Yingliang and Yu, Jingyi and Shi, Yujiao},
  journal={arXiv preprint arXiv:2603.17571},
  year={2026}
}
```

## License

This project is released under the MIT License. See the [LICENSE](https://www.google.com/search?q=./LICENSE) file for details about the license under which this code is made available.
