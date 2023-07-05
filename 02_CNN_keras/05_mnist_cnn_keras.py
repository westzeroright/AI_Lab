#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# mnist 데이터 가져오기
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)

# X값의 타입을 float형으로 변환
x_train = tf.cast(x_train,dtype=tf.float32)
print(x_train.shape,x_train.dtype)

x_test = tf.cast(x_test,dtype=tf.float32)
print(x_test.shape,x_test.dtype)


# In[3]:


## CNN 사용 안 한 모델
x_train = tf.reshape(x_train,[-1,28*28]) # 2차원
x_test = tf.reshape(x_test,[-1,28*28]) # 2차원
print(x_train.shape,x_train.dtype)
print(x_test.shape,x_test.dtype)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=32,activation='relu',input_shape=(784,)),
    tf.keras.layers.Dense(units=32,activation='relu'),
    tf.keras.layers.Dense(units=32,activation='relu'),
    tf.keras.layers.Dense(units=10,activation='softmax'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.summary()


# In[4]:


# 학습
model.fit(x_train,y_train,epochs=100,batch_size=32,verbose=1)


# In[5]:


model.evaluate(x_test,y_test)


# In[16]:


# CNN을 사용한 모델 구현
x_train = tf.reshape(x_train,[-1,28,28,1]) # 2차원
x_test = tf.reshape(x_test,[-1,28,28,1]) # 2차원
print(x_train.shape,x_train.dtype)
print(x_test.shape,x_test.dtype)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',
                          activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'),
    
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',
                          activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'),
    
    tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',
                          activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512,activation='relu'),
    tf.keras.layers.Dense(units=10,activation='softmax'),

])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.summary()

# https://yeomko.tistory.com/40
# Xavier Glorot Initialization : W(Weight) 값을 fan_in,fan_out를 사용하여 초기화하여 정확도 향상

# loss 종류
# mean_squared_error : 평균제곱 오차
# binary_crossentropy : 이진분류 오차
# categorical_crossentropy : 다중 분류 오차. one-hot encoding 클래스, [0.2, 0.3, 0.5] 와 같은 출력값과 실측값의 오차값을 계산한다.
# sparse_categorical_crossentropy: 다중 분류 오차. 위와 동일하지만 , integer type 클래스라는 것이 다르다.


# In[18]:


# 학습 : 약 2분 소요(GPU)
model.fit(x_train,y_train,epochs=25,validation_split=0.25)


# In[19]:


# 평가
model.evaluate(x_test,y_test)


# ### 모델 개선

# In[26]:


# VGGNet (VGG-19) 스타일의 MNIST 분류 CNN 모델 
#--------------------------------------------
# ( Conv2D * 2개  --> MaxPool2D ) * 2회 : 4층
# ( Conv2D * 4개  --> MaxPool2D ) * 3회 : 12층
# Dense * 3개                           : 3층
#--------------------------------------------
#                                     총 19층
#--------------------------------------------
# 각 네트워크마다 필터의 수를 2배로 증가 시킨다 : 32-->64-->128-->256-->512

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',
                          activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',
                          activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',
                          activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding='valid',
                          activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    
    
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=256,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10,activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.summary()


# In[27]:


# 학습 : 약 4분 소요(GPU)
model.fit(x_train,y_train,epochs=25,validation_split=0.25)


# In[28]:


# 평가
model.evaluate(x_test,y_test)


# In[ ]:




