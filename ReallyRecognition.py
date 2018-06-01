
# coding: utf-8

# In[1]:


#匯入所需函數
import cv2, sys, json, time, numpy as np, tensorflow as tf
from pygame import mixer
#cascPath = "/home/test/anaconda3/pkgs/opencv3-3.2.0-np111py35_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
from train import load_model, Model

#result接收人臉辨識辨識結果並查詢audioList播放我們透過mensTalk函數錄好的恭喜發財錄音檔
#t0;t1;waitTime是控制播放間隔參數
t0=0;t1=0;waitTime=5;result = 0
audioList = ['其他人','孫爺爺','發哥','志玲姊姊','范爺','阿妹','阿武','蔡姊']

#初始化我們開發的人臉"分類"AI
model = Model()
model.load()
#初始化音檔撥放功能
mixer.init()

# In[3]:

# 攝影機連線
cap = cv2.VideoCapture(0)

while(True):
  # 永真函數
  print(cap.isOpened())
  try :
    # 參數ret的值為True或False，代表有沒有讀到圖片，參數frame，是當前截取一幀的圖片
    ret, frame = cap.read()
    # Opencv內建的人臉辨識函式使需要灰階圖片故用cvtColor將BGR轉GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  except :
    print("")
  #opencv3內建人臉識別功能函數
  faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE  #remove in opencv3
    flags = cv2.CASCADE_SCALE_IMAGE
  )
  #將辨識出來的人臉透過for迴圈取出
  for (x, y, w, h) in faces:
    #標出人臉並畫線
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
    image = frame[y - 70: y + h+70, x-20: x + w+20]
    
    #因為音檔有八秒鐘故使用t0、t1、waitTime三個參數控制分類頻率，
    t0 = int(time.time())
    if image is None :
        print("image is None")
    else :
      try :
           image = cv2.resize(image  ,(450,450),interpolation=cv2.INTER_CUBIC)
           #顯示圖片，看看顏色大小人臉辨識功能是否正常
           cv2.imshow('frame', image)   
           #因Opencv預設格式是BGR必須轉成RGB
           image = image[:,:,::-1]
       
      except :
           print(" error:", sys.exc_info()[0])
      else :
           #圖片傳入我們開發的人臉"分類"AI並得到結果
           result = model.predict(image)

      #如果辨識結果不為0(nobody)，且大於waitTime則播放對應的音檔
      if result != 0 and t0-t1 > waitTime : 
           #撥放對應音檔
           videoPath = "./audio/"+audioList[result]+".mp3"
           print (audioList[result]+" "+"Happy new year")
           mixer.music.load(videoPath)
           mixer.music.play()
           t1=int(time.time())

  # 若按下 q 鍵則離開迴圈
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()

