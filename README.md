# UniSyn:A Multi-Modal Framework with Knowledge Transfer for Drug Synergy Prediction<br/>
This repository contains the source code for the paper
![image](./model.png)
<br/>
Please see our manuscript for more details.<br/>
### Installation
1. Clone the repository.

   ```python
   git clone https://github.com/codejiajia/UniSyn.git
   ```
2. Create a virtual environment by conda.

   ```python
   conda create -n UniSyn_env python=3.9.20
   conda activate UniSyn_env
3. Download PyTorch>=2.4.1, which is compatible with your CUDA version and other Python packages.

   ```python
   conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -c pytorch # for CUDA 12.1
   pip install -r requirements.txt
   ```
### Data and model checkpoints
The following data and model checkpoints are available at [zenodo](https://zenodo.org/records/16352141).
