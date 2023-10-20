# AI_SIMPLE_CUDA

AI_SIMPLE_CUDA is a neural network framework written in C++, designed to leverage both CPU and GPU compute capabilities. The framework currently supports Multi-Layer Perceptrons (MLPs) and includes preliminary, non-functional code for Convolutional Neural Networks (ConvNets). The primary goal of AI_SIMPLE_CUDA is to provide a testing ground for comparing its performance and accuracy against other common neural network frameworks.

## Description

AI_SIMPLE_CUDA is designed to provide a simple, yet powerful, interface for developing and training neural networks in C++. The framework is built with a focus on flexibility and ease of use, with a clean and simple API that makes it easy to create complex neural network architectures. It is designed to take full advantage of both CPU and GPU compute capabilities, and is currently compatible with Ubuntu.

## Features

- **Multi-Layer Perceptrons (MLPs)**: Build and train MLPs with ease, using a simple and intuitive API.
- **Convolutional Neural Networks (ConvNets)**: Preliminary, non-functional code for ConvNets is included. This feature is currently under development.
- **Tensor Implementation**: The framework uses a custom Tensor implementation, which is essentially a 3-dimensional array. Currently, the number of dimensions is fixed at 3.
- **GPU Preloading**: To maximize speed, data is preloaded into the GPU. This allows for faster processing and better utilization of the GPU's capabilities.
- **Parallelization**: The framework is designed to parallelize tasks as much as possible, further enhancing performance.

## Installation

Follow these commands below to properly set up for ubuntu. Be sure to download the proper cuda version for ubuntu

More version can be found here:
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/

### ubuntu20.04 
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
```

### ubuntu22.04
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin

sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
```

Follow these steps for all versions
```
sudo apt-get update

sudo apt-get -y install cuda
```

### Installing ImageMagick
```
https://techpiezo.com/linux/install-imagemagick-in-ubuntu-20-04-lts/

ImageMagick

git clone https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.0
cd ImageMagick-7.1.0
sudo apt-get install -y libtiff-dev
sudo apt-get install -y libjpeg62-dev 
sudo apt-get install -y libpng-dev
sudo apt install -y zlib1g
sudo apt install -y zlib1g-dev
./configure
make

#If build fails, try gmake instead.

sudo make install
sudo ldconfig /usr/local/lib
make check
```

cd ..


// Not necessary anymore because submodules: `git clone git@github.com:Tacosaurus100/coreutils.git`

// Not necessary anymore because submodules: cd AI_Simple_CUDA

// Not necessary anymore because submodules: ln -s ../coreutils/ ./

// more docs for installing cuda
https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#linux

// this is how to get your dpkg stuff
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda

sudo reboot
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

## Usage

See testing folder for examples of each part.