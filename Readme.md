# Benchmarking IBM Analog Hardware KIT performance for different algorithms

Goal/Objective - The goal is to implement and benchmark standard deep
Learning algorithms and do performance profiling with different setting
such as Single/multi/distributed training, and different tiling
configuration using the IBM AIHWKIT.

Challenges - Current repo has a certain implementation that shows the
result with some algorithm, however, the model used is not standard
hence difficult to benchmark its performance and the speed-up the
toolkit provides. Also, there’s no support for distributed training setting

Approaches/ Techniques – We initially intend to understand the Resnet-18
model and perform its benchmark in PyTorch on CIFAR-10 and create it
Analog version and experiment with different hardware simulations such as
Single RPU, ReRam, etc.

Implementation details - Setting up the necessary environment locally and
then replicating it on the HPC environment for the Analog-KIT library.
• Hardware – TeslaV100/RTX8000 GPU.
• Framework – Python, Pytorch, AIHWKIT.

## Authors

It's a joint work between Neel and Vijay.

- [Neel Shah](https://www.github.com/deadpanther)
- [Vijayraj Gohil](https://www.github.com/vraj130)


## Setup

The library is compatible with Python Version 3.7

```
# To setup the library you need to have anaconda installed.
module load anaconda3/<VERSION_NUMBER>
#We have set up a singularity environment
singularity exec --overlay /home/<USER>/<YOUR SINGULARITY CONTAINER NAME> /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash
#Now we are inside the singularity container
#Setup the conda bash in the container
bash /share/apps/utils/singularity-conda/setup-conda.bash
#We now source the environment file
source /ext3/env.sh
#Install the required libraries then
conda install -y cmake openblas pybind11
conda install -y -c conda-forge scikit-build
#Install the Torch version for this.
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install scipy protobuf
#Install the aihwkit library with the CUDA installation flag on
pip install -v aihwkit --install-option="-DUSE_CUDA=ON”
```

Another alternative to install the library is:
```
conda create -n myenv python=3.7
conda activate myenv
#Install the required libraries then
conda install -y cmake openblas pybind11
conda install -y -c conda-forge scikit-build
#Install the Torch version for this.
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install scipy protobuf
#Install the aihwkit library with the CUDA installation flag on
pip install -v aihwkit --install-option="-DUSE_CUDA=ON”
```
Create a virtual environment to install project dependencies.

    
## Usage
The required scripts to run the models in normal mode with 1 GPU and also with multiple nodes and multiple GPUs is provided in the scripts section.