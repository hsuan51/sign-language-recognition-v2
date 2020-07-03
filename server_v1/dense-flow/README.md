# Dense-Flow
---
## 安裝OpenCV環境
### setup
https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/
```linux
sudo apt-get upgrade
sudo apt-get install build-essential cmake unzip pkg-config libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python-dev
```
### 下載OpenCV
```linux
sudo wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.4.zip
sudo unzip opencv.zip
sudo mv opencv-3.4.4 opencv
sudo wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.4.zip
sudo unzip opencv_contrib.zip
sudo mv opencv_contrib-3.4.4 opencv_contrib
cd ~/opencv
sudo mkdir build
cd build
sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/opencv/ -D WITH_CUDA=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D BUILD_EXAMPLES=ON -D WITH_FFMPEG=ON -D CUDA_ARCH_BIN=6.0 -DBUILD_opencv_cudacodec=OFF ..
sudo make -j4
sudo make install
```
查看GPU型號的[CUDA_ARCH_BIN](https://www.cnblogs.com/beihaidao/p/6773595.html)

---
## 提取光流
### 1.修改build-flow.sh LD_LIBRARY_PATH路徑
### 2.找出影片路徑並生成txt檔 
```
find /path/to/your/data -name "*.mp4" > files.txt
```
data為影片資料夾
### 3.運行批次檔提取光流
```shell=
sudo bash build-flow.sh
```

---
原作者[https://github.com/agethen/dense-flow](https://github.com/agethen/dense-flow)

---
