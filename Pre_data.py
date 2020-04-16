import cv2
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
import random
import numpy as np
import os

filepath=''
def load_img(path,grayscale=False):
    if grayscale:
        img=cv2.imread(path)
    else:
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=np.array(img,dtype='float')/255
    return img
def get_train_val(val_rate=0.2):
    train_url=[] #每张图片的路径
    train_set=[]
    val_set=[]
    for i in os.listdir(filepath+'src'):
        train_url.append(i) ##名字 3.png
    random.shuffle(train_url) ##打乱
    total_num=len(train_url)
    val_num=int(val_rate*total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set,val_set

def generateData(batch_size,data=[]):##给生成器分批次产生训练数据
    while True:
        train_data=[]
        train_label=[]
        batch=0
        for i in range(len(data)):
            url=data[i]
            batch+=1
            img=load_img(filepath+'src/'+url)
            img=img_to_array(img)
            train_data.append(img)
            label=load_img(filepath+'label/'+url,grayscale=True)
            label=img_to_array(label)
            train_label.append(label)
            if batch % batch_size==0:
                train_data=np.array(train_data)
                train_label=np.array(train_label)
                yield (train_data,train_label) ### 返回值同时下次从程序下一行开始，即赋值0
                train_data=[]
                train_label=[]
                batch=0

def generateVal(batch_size,data=[]):##给生成器分批次产生验证数据
    while True:
        val_data=[]
        val_label=[]
        batch=0
        for i in range(len(data)):
            url=data[i]
            batch+=1
            img=load_img(filepath+'src/'+url)
            img=img_to_array(img)
            val_data.append(img)
            label=load_img(filepath+'label/'+url,grayscale=True)
            label=img_to_array(label)
            val_label.append(label)
            if batch % batch_size==0:
                train_data=np.array(val_data) #批次的数组
                train_label=np.array(val_label)
                yield (train_data,train_label) ### 返回值同时下次从程序下一行开始，即赋值0
                train_data=[]
                train_label=[]
                batch=0
'''
callable=ModelCheckpoint(
                            
                            )加入输出路径等等
model.fit_generator(generator=   ,steps_per_epoch=   ,epochs=   ,verbose=1,  
                    validation_data=    ,validation_steps=   ,callbacks=callable,max_q_size=1)  
                    steps_per_epoch  == train_num/batch
                    validation_steps == val_num/batch
'''

