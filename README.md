# 基於3D卷積神經網絡之手語翻譯系統
>在前端畫面提取影像光流時使用了<https://github.com/agethen/dense-flow>  
>而後端的3D-CNN模型則是使用<https://github.com/LossNAN/I3D-Tensorflow>

首先在GCP上建立一個VM，作業系統為UBUNTU 18.04，並下載一些相依的套件。
## 安裝cuda
先加入nvidia的repository
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update
```

下載nvidai的driver
```
sudo apt-get install --no-install-recommends nvidia-driver-430
```

重新啟動VM之後確認nvidia驅動已經下載完成
```
nvidia-smi
```

下載development與runtime libraries(約4GB)
```
sudo apt-get install --no-install-recommends \
    cuda-10-1 \
    libcudnn7=7.6.4.38-1+cuda10.1  \
    libcudnn7-dev=7.6.4.38-1+cuda10.1
```

下載TensorRT
```
sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1
```

之後檢查在自己的環境下的Python版本，並下載pip
```
sudo apt-get install python #下載python
sudo apt install python-dev python-pip #下載pip及python套件
python --version
pip --version
```

之後把整個專案git下來
```
git clone https://github.com/hsuan51/sign-language-recognition-v2.git
```

下載提取光流的相依套件。
```
sudo apt-get upgrade
sudo apt-get install build-essential cmake unzip pkg-config libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python-dev
```
---

## 下載opencv
```
sudo wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.4.zip
sudo unzip opencv.zip
sudo mv opencv-3.4.4 opencv
sudo wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.4.zip
sudo unzip opencv_contrib.zip
sudo mv opencv_contrib-3.4.4 opencv_contrib
cd ~/opencv
sudo mkdir build
cd build
```

查看自己gpu型號的[CUDA_ARCH_BIN](https://developer.nvidia.com/cuda-gpus)並將CUDA_ARCH_BIN參數改為自己相應的數值
```
sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/opencv/ -D WITH_CUDA=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D BUILD_EXAMPLES=ON -D WITH_FFMPEG=ON -D CUDA_ARCH_BIN=3.7 -D BUILD_opencv_cudacodec=OFF ..
```
```
sudo make -j2
sudo make install
```
測試dense-flow
進入/server_v1/dense-flow
先編輯Makefile.config
```
INCLUDE=-I/some/path/opencv/include -Iinclude/
LIB=-L/some/path/opencv/lib
```
進入/etc/ld.so.conf.d
```
sudo nano opencv.conf
```
加入/usr/local/opencv/lib
測試提取光流
```
./denseFlow_gpu --vidFile="video.mp4" --xFlowFile="flow_x" --yFlowFile="flow_y" --imgFile="im" --bound=16 --type=2 --device_id=0 --step=10
```
[下載nginx](https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-18-04)
[在no-ip申請免費domain](https://www.noip.com/)
將client_v1移至/var/www/html
```
sudo mv client_v1/ /var/www/html/
```
[利用Let's Encrypt加入SSL](https://www.digitalocean.com/community/tutorials/how-to-secure-nginx-with-let-s-encrypt-on-ubuntu-18-04)
下載一些Python的相依套件
```
pip install --upgrade setuptools
sudo apt-get install protobuf-compiler python-pil python-lxml
sudo pip install jupyter matplotlib opencv-python==3.2.0.8 contextlib2 dm-sonnet==1.23
sudo pip install flask
sudo pip install flask-cors
```




