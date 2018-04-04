# HappyNewYear.Ai :辨識您的親朋好友，並自動向他們祝賀新年快樂
Author:Ray.Tseng
# 環境部屬
#ubuntu16.04
#安裝Anaconda  
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh  
bash Anaconda3-5.0.1-Linux-x86_64.sh  

#建立virtualenv  
conda create -n HappyNewYear python=3.5  
source activate HappyNewYear  

#安裝package  
conda install -c https://conda.anaconda.org/menpo opencv3  
conda install -c anaconda pyqt=4.11.4  
sudo apt-get install libgtk2.0-0 git python-wxtools  
git clone https://github.com/dataisfunny/HappyNewYear.AI.git  
cd  HappyNewYear.AI  
pip install -r requirements.txt  

#調整Keras參數
mkdir ~/.keras  
cat >> ~/.keras/keras.json << EOF
{
    "backend": "tensorflow",
    "epsilon": 1e-07,
    "image_data_format": "channels_first",
    "floatx": "float32"
}
EOF
    
#(可選)設定jupyter   
jupyter-notebook --generate-config  
sed -i 's/#c.NotebookApp.allow_password_change/c.NotebookApp.allow_password_change/g' ~/.jupyter/jupyter_notebook_config.py  

jupyter notebook password  
jupyter notebook --ip=0.0.0.0 --port=8888  

# 啟動程式
#Streaming模式(影像從相機來)  
sudo python imageStreaming.py  
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/A3Z6aCFrGIo/0.jpg)](http://www.youtube.com/watch?v=A3Z6aCFrGIo)

#imageFile模式(影像從檔案來)  
sudo python imageFile.py  
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/iN_6Ya-e-IM/0.jpg)](http://www.youtube.com/watch?v=iN_6Ya-e-IM)

# 訓練模型
1.按照data/train中的格式，要分幾類就開幾個資料夾，Keras imageGenerator會自動檔把資料夾名稱當類別名稱  
2.刪掉已經訓練完的神經網路 faces6.h5  
3.sudo python train.py  

# 參考  
1.BossSensor (經典)  
2.Video Stream Processor (結合大數據打造高擴張的AI應用)  
3.building-powerful-image-classification-models-using-very-little (基礎)




