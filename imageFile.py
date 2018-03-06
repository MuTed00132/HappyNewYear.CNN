
# coding: utf-8

# In[1]:


import tensorflow as tf
import cv2, sys, json, time, numpy as np, wx
from pygame import mixer
cascPath = "/home/ray/anaconda3/pkgs/opencv3-3.2.0-np111py35_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
from train import load_model
from train import Model
from PyQt4 import QtCore, QtGui

# 自定義的窗口類別
class ImageWindow(QtGui.QWidget):
    count = 0
    audioList = ['其他人','孫爺爺','發哥','志玲姊姊','范爺','阿妹','阿武','蔡姊']
    # 初始化
    def __init__(self, parent = None):
        super(ImageWindow, self).__init__(parent)
        self.setWindowTitle(u'Ray Bulding')

        # 創建按鈕UI
        self.pushButton = QtGui.QPushButton(u'Next')

        # 創建Lebal UI
        self.LebalUI = QtGui.QLabel()
        
        self.LebalUI.setPixmap(QtGui.QPixmap("./temp/python.jpeg"))
  
        # 創建垂直Layout
        layout = QtGui.QVBoxLayout()

        # 將UI放到Layout
        layout.addWidget(self.LebalUI)
        layout.addWidget(self.pushButton)

        # 交Layout變數丟給setLayout function
        self.setLayout(layout)

        # 設置按鈕點擊事件並將要執行的函數(sayHappyNewYear)當參數丟進去
        self.pushButton.clicked.connect(self.sayHappyNewYear)

    # 編輯sayHappyNewYear函數
    def sayHappyNewYear(self):
        #創造AI類別 instance
        model = Model()
        #載入深度學習後的殘餘矩陣
        model.load()
        #將全域變數+1後丟給區域變數
        self.count= self.count+1
        count=str(self.count)
        #透過變數將圖片以OpenCV的函數取出並重置像素
        show = "./data/test/"+count+".jpg"
        frame = cv2.imread(show)
        temp = cv2.resize(frame ,(300,300),interpolation=cv2.INTER_CUBIC)
        #並將其中一份暫存到temp.jpg給QT的圖片顯示函數使用
        cv2.imwrite("./temp.jpg", temp)
        #QT的圖片顯示函數去取剛剛透過OpenCV處理完的圖片
        self.LebalUI.setPixmap(QtGui.QPixmap("./temp/temp.jpg"))
        #將一開始讀到的Raw Image(frame)丟到OpenCV內建的人臉識別演算法，這套演算法只需使用灰階圖片故使用cvtColor轉置一份灰階圖片出來
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #OpenCV內建的人臉識別演算法
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE)
        #透過For迴圈將識別完的臉譜取出，因OpenCV是以BGR儲存圖片故要先轉置成RBG再丟到我們訓練的人臉分類神經網路，最後啟動對應的新年快樂MP3
        for (x, y, w, h) in faces:
            #框出臉
            image=cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
            #舊參數#image = frame[y - 70: y + h+70, x-20: x + w+20]
            #BGR轉RGB
            image = image[:,:,::-1]
            #將臉譜丟到丟到我們訓練的人臉分類神經網路
            result=model.predict(image)
            #啟動對應的新年快樂MP3
            videoPath = "./audio/"+self.audioList[result]+".mp3"
            mixer.init()
            mixer.music.load(videoPath)
            mixer.music.play()   
        
# 如果是單獨執行
if __name__=='__main__':
    
    #QT啟動
    app = QtGui.QApplication(sys.argv)
    mainWindow = ImageWindow()
    mainWindow.show()
    sys.exit(app.exec_())
    
    

