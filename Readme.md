# Benchmarking IBM Analog Hardware KIT performance for different algorithms

## What is Analog AI?

In traditional hardware architecture, computation and memory are siloed in different locations. Information is moved back and forth between computation and memory units every time an operation is performed, creating a limitation called the von Neumann bottleneck.



Analog AI delivers radical performance improvements by combining compute and memory in a single device, eliminating the von Neumann bottleneck. By leveraging the physical properties of memory devices, computation happens at the same place where the data is stored. Such in-memory computing hardware increases the speed and energy-efficiency needed for next generation AI workloads.


## What is an In-Memory computing chip?

An in-memory computing chip typically consists of multiple arrays of memory devices that communicate with each other. Many types of memory devices such as phase-change memory (PCM), resistive random-access memory (RRAM), and Flash memory can be used for in-memory computing.

Memory devices have the ability to store synaptic weights in their analog charge (Flash) or conductance (PCM, RRAM) state. When these devices are arranged in a crossbar configuration, it allows to perform an analog matrix-vector multiplication in a single time step, exploiting the advantages of analog storage capability and Kirchhoff’s circuits laws. You can learn more about it in our online demo.

In deep learning, data propagation through multiple layers of a neural network involves a sequence of matrix multiplications, as each layer can be represented as a matrix of synaptic weights. The devices are arranged in multiple crossbar arrays, creating an artificial neural network where all matrix multiplications are performed in-place in an analog manner. This structure allows to run deep learning models at reduced energy consumption.

## Which library can be used to simulate in-memory computing with analog devices?

IBM Research have come up with a library which can be used to simulate in-memory computing with analog devices. They have many analog device models available, and you can use them to simulate in-memory computing. The library havs several different tiling configurations for different types of memory devices. The library also has a set of functions to simulate the behavior of the devices. The library aims to mimic the behavior of the analog devices in the real world. 
*Note*- The IBM AIHWKIT is currently in development phase and has many inconsistent implementations which makes it a task in itself to install and run.

## What is the Goal and objectives of our project?

The goal is to implement and benchmark standard Deep Learning algorithms and do performance profiling with different setting such as Single or Multi in a traditional training and distributed training, and using different configurations for the Analog memory devices using the IBM AIHWKIT. We try to benchmark the performances of the models using a single node and multiple nodes as well. We also try to benchmark the performance of the models using different configurations of the Analog memory devices. This enables us to compare the performance of the models using different configurations of the Analog memory devices. We try to write consistent code with a high level of abstraction and modularity to make it easy to use as well as to make it easy to extend.

## What are the challenges faced while using the library and how we overcame them?
Current repo has a certain implementation that shows the result with simple Le-Net5 architecture, however, the model used is not standard hence difficult to benchmark its performance and the speed-up the toolkit provides. Also, there’s no support for distributed training setting.
It is a simulation of the analog devices and tries to mimick the performances on real-world analog devices and hence it is not that reliable right now.

## What approaches are used to benchmark the performance of the algorithms?
We initially intend to understand the Resnet-18 model and perform its benchmark in PyTorch on CIFAR-10 and create it Analog version and experiment with different hardware simulations such as Single RPU, ReRam, etc. We used Python's benchmarking tools to benchmark the performance of the models.

## What is the hardware and software used for the project?
Setting up the necessary environment locally and then replicating it on the HPC environment for the Analog-KIT library.
The working of the repository is all possible due to the help of NYU's HPC cluster and its wonderful team who have helped us in setting it up.
• Hardware – TeslaV100 and RTX8000 GPU.
• Framework – Python(v3.7.13), Pytorch(v1.8.1+cu111), AIHWKIT library(v0.5.1).

## Authors

It's a joint work between Neel and Vijay.

- [Neel Shah](https://www.github.com/deadpanther)
- [Vijayraj Gohil](https://www.github.com/vraj130)


## Setup

The library is compatible with Python Version 3.7

```
# To setup the library you need to have anaconda installed.
module load anaconda3/<VERSION_NUMBER>
#Load the cuda module
module load cuda/11.3.1

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


## References

Malte J. Rasch, Diego Moreda, Tayfun Gokmen, Manuel Le Gallo, Fabio Carta, Cindy Goldberg, Kaoutar El Maghraoui, Abu Sebastian, Vijay Narayanan. "A flexible and fast PyTorch toolkit for simulating training and inference on analog crossbar arrays" (2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems)