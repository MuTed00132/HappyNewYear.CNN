
# coding: utf-8

# In[6]:


import os,cv2
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

for root, dirs, files in os.walk("./data/train", topdown=False):
    i=0
    j=0
    for name in files:
        frame = cv2.imread(os.path.join(root, name))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE)
        i=i+1
        j=0
        for (x, y, w, h) in faces:
            image = cv2.resize(image,(600,600),interpolation=cv2.INTER_CUBIC)
            image = frame[y-30 : y + h+30, x-20: x + w+20]
            fname = str(i) + str(j) +"."+ name.split('.', 1 )[1]
            path=os.path.join(root,fname)
            cv2.imwrite(path, image)
            
            j=j+1
            image = frame[y-30 : y + h+60, x-40: x + w+40]
            fname = str(i) + str(j) +"."+ name.split('.', 1 )[1]
            path=os.path.join(root,fname)
            cv2.imwrite(path, image)
            
            j=j+1
            image = frame[y : y + h, x: x + w]
            fname = str(i) + str(j) +"."+ name.split('.', 1 )[1]
            path=os.path.join(root,fname)
            cv2.imwrite(path, image)


# In[ ]:


fname = name
print (name.split('.', 1 )[1]);

