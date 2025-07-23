# UniSyn:A Multi-Modal Framework with Knowledge Transfer for Drug Synergy Prediction<br/>
This repository contains the source code for the paper
![image](./model.png)
<br/>
Please see our manuscript for more details.<br/>
## Installation
1. Clone the repository.

   ```python
   git clone https://github.com/codejiajia/UniSyn.git
   ```
2. Create a virtual environment by conda.

   ```python
   conda create -n UniSyn_env python=3.8.20
   conda activate UniSyn_env
3. Download PyTorch>=1.12.1, which is compatible with your CUDA version and other Python packages.

   ```python
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch # for CUDA 11.3
   pip install -r requirements.txt
   ```
