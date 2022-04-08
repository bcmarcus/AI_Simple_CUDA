
Required libraries
cuda 11.6

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

sudo apt remove cuda

ImageMagick

git clone https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.0
cd ImageMagick-7.1.0
./configure
make

#If build fails, try gmake instead.

sudo make install
sudo ldconfig /usr/local/lib
make check