# Train I3D model
---
## 安裝Tensorflow GPU
環境為ubuntu18.04
### Setup
```linux
sudo apt-get update
sudo apt-get install unzip python-pip python-dev protobuf-compiler python-pil python-lxml
sudo pip install --upgrade pip
sudo pip install jupyter matplotlib opencv-python==3.2.0.8 contextlib2 dm-sonnet==1.23
```
### 安裝顯示卡驅動
```linux
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update
sudo apt-get install --no-install-recommends nvidia-driver-418
```
安裝完後務必重啟VM
```linux
nvidia-smi
```
重啟後查看是否安裝成功
### 安裝CUDA&CUDNN
```linux
sudo apt-get install --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.6.2.24-1+cuda10.0  \
    libcudnn7-dev=7.6.2.24-1+cuda10.0
sudo apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 \
    libnvinfer-dev=5.1.5-1+cuda10.0
```
### 安裝Tensorflow-GPU
```linux
sudo pip install tensorflow-gpu
```
---
## Train I3D
### 1.利用DenseFlow提取光流
[https://github.com/asd17539/dense-flow](https://github.com/asd17539/dense-flow)

### 2.將資料及轉為train.list與test.list
```linux
cd ./list/ucf_list/
sudo bash ./convert_images_to_list.sh /path/to/video_data
```
### 3.修改checkpoint的路徑
```linux
cd ./checkpoints/
sudo nano ./flow_imagenet/checkpoints
sudo nano ./flow_scratch/checkpoints
sudo nano ./rgb_imagenet/checkpoints
sudo nano ./rgb_scratch/checkpoints
```
將路徑修改
### 4.Training model
選擇Pre_Train的Model類型 (line 46 ~ 47)
設置Training Model的名稱 (line 49 ~ 50)
```linux
cd ./experiments/ucf-101
sudo python train_ucf_rgb.py
sudo python train_ucf_flow.py
```
### 5.Testing model
選擇Training Model
```linux
cd ./experiments/ucf-101
sudo python test_ucf_rgb.py
sudo python test_ucf_flow.py
sudo python test_ucf_rgb+flow.py
```

---
原作者[https://github.com/LossNAN/I3D-Tensorflow](https://github.com/LossNAN/I3D-Tensorflow)

---
