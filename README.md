
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/

{

// ubuntu20.04 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

// ubuntu22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin

sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

}

sudo apt-get update

sudo apt-get -y install cuda


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

cd ..

git clone git@github.com:Tacosaurus100/coreutils.git

cd AI_Simple_CUDA

ln -s ../coreutils/ ./


// these are the docs
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