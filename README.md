<div align="center">

<h1>
<img src="assets/HemiDiff.png" height="40" style="vertical-align:middle; margin-right:10px;">
Hemispheric Diffusion
</h1>

<b>A Compositional Generative Policy for Coordinated Bimanual Manipulation</b>

</div>

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://yechen056.github.io/HemiDiff/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg)](https://pytorch.org/)

[Yechen Fan](https://github.com/yechen056)<sup>1,2*</sup>, [Jinhua Ye](#)<sup>1,2*</sup>, [Xinyou Ji](#)<sup>1</sup>, [Haibin Wu](#)<sup>1</sup>, [Gengfeng Zheng](#)<sup>2†</sup> and [Huixin He](#)<sup>3†</sup>

<sup>1</sup> Fuzhou University &nbsp;&nbsp; <sup>2</sup> Fujian Key Laboratory of Special Intelligent Equipment Safety Measurement and Control &nbsp;&nbsp; <sup>3</sup> Huaqiao University

*Equal contribution &nbsp;&nbsp; † Corresponding authors*

**[[Project Page](https://yechen056.github.io/HemiDiff/)] | [[Paper](#)]**

</div>

<div align="center">
  <img src="assets/teaser.jpg" alt="HemiDiff teaser" width="90%" />
</div>

<p align="justify">Hemispheric Diffusion enhances robustness by using an asymmetric routing mechanism to mask failed modalities and dynamically adjust multimodal contributions without retraining, unlike standard early-fusion strategies.</p>

## 📖 Overview

**Hemispheric Diffusion** is a bimanual control framework that addresses modality dominance (vision suppressing tactile cues) while enforcing physical coordination between two arms. 

- **Asymmetric Perception Router:** Dynamically reweights left/right perceptual streams according to occlusion and contact states.
- **Independent Modality Experts:** Supports sensor masking at inference time (e.g., broken camera or tactile stream) without retraining.
- **Coordination Energy Expert:** Adds geometric consistency during diffusion denoising to avoid collision and desynchronization.

<div align="center">
  <img src="assets/simulation.gif" width="100%" alt="simulation">
</div>

---

## 🛠️ Installation

<details>
<summary><b>Click to installation steps</b></summary>

### 1. Create Conda Environment

```bash
conda create -n hemidiff python=3.9 -y
conda activate hemidiff
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

git clone --recursive https://github.com/yechen056/Hemispheric-Diffusion.git
cd Hemispheric-Diffusion

pip install -r requirements.txt
pip install -e .
```

### 2. Install CoppeliaSim 4.1

```bash
mkdir -p ~/.coppeliasim
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C ~/.coppeliasim --strip-components 1
```

Add to `~/.bashrc`:

```bash
export COPPELIASIM_ROOT=${HOME}/.coppeliasim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Then reload:

```bash
source ~/.bashrc
```

### 3. Install Third-Party Packages

```bash
cd third_party/PyRep && pip install -e .
cd ../RLBench && pip install -e .
cd ../YARR && pip install -e .
cd ../pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install -e .
cd ../../
```
</details>

---

## 🚀 Quick Start

### 🗂️ 1. Generate Demonstrations

```bash
python scripts/gen_rlbench_bimanual_data.py gen rlbench -t coordinated_push_box -c 100
```

Arguments:
- `-t`: Task name (e.g., `coordinated_push_box`)
- `-c`: Number of episodes

Supported simulation tasks:
- `bimanual_straighten_rope`
- `coordinated_lift_ball`
- `coordinated_lift_tray`
- `coordinated_push_box`
- `coordinated_put_item_in_drawer`
- `dual_push_buttons`

### 🚆 2. Train

```bash
python scripts/train.py \
  task=rlbench/coordinated_push_box \
  task.dataset.zarr_path=data/rlbench/coordinated_push_box_expert_100.zarr
```

### 🎯 3. Evaluate

```bash
python scripts/eval.py \
  -c output/checkpoints/latest.ckpt \
  -o output/eval \
  -n 100
```

💡 Rollout videos are saved to `output/eval/media/`.

---

<!-- ## 📝 Citation

```bibtex
@article{fan2025hemidiff,
  title={Hemispheric Diffusion: A Compositional Generative Policy for Coordinated Bimanual Manipulation},
  author={Fan, Yechen and Ye, Jinhua and Ji, Xinyou and Wu, Haibin and Zheng, Gengfeng and He, Huixin},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
``` -->

## 📄 License

This project is released under the [MIT License](LICENSE).

## 🙏 Acknowledgements

This work builds upon excellent open-source projects including [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [RLBench](https://github.com/stepjam/RLBench), [PyRep](https://github.com/stepjam/PyRep), and [Perceiver-Actor^2](https://github.com/markusgrotz/peract_bimanual). We thank the authors and maintainers for their contributions.
