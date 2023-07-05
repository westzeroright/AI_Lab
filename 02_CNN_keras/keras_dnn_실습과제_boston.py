#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Boston_Dataset_Linear Regression
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
tf.random.set_seed(5)

boston_train = np.loadtxt('boston_train.csv', delimiter=',', dtype=np.float32, skiprows=1)
boston_test = np.loadtxt('boston_test.csv', delimiter=',', dtype=np.float32, skiprows=1)

x_train = boston_train[:,:-1]
y_train = boston_train[:,[-1]]
print(x_train.shape)
print(y_train.shape)

x_train = np.array(x_train,dtype=np.float32)
y_train = np.array(y_train,dtype=np.float32)


# In[2]:


# Dense Layer 구현 : 2층 구현
model = tf.keras.Sequential([
    # 첫번째 층 출력 : [None,20],   활성화 함수 : 'relu'
    tf.keras.layers.Dense(units=20,activation='relu',input_shape=(9,)) ,
    # 두번째 층 출력 : [Non,10],   활성화 함수 : 'relu'
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='mean_squared_error')

model.summary()


# In[3]:


# 학습
history = model.fit(x_train,y_train,epochs=700,batch_size=1,verbose=1) # verbose=1, 메세지를 출력


# In[4]:


# 시각화
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[5]:


# 예측
model.predict(x_train)


# In[6]:


# 평가
x_test = boston_test[:,:-1]
y_test = boston_test[:,[-1]]
model.evaluate(x_test,y_test)


# In[7]:


# Dense Layer 구현 : 3층 구현
model = tf.keras.Sequential([
    # 첫번째 층 출력 : [None,20],   활성화 함수 : 'relu'
    tf.keras.layers.Dense(units=20,activation='relu',input_shape=(9,)) ,
    # 두번째 층 출력 : [Non,10],   활성화 함수 : 'relu'
    tf.keras.layers.Dense(units=10),
    # 세번째 층 출력 : [Non,10],   활성화 함수 : 'relu'
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='mean_squared_error')

model.summary()


# In[8]:


# 학습
history = model.fit(x_train,y_train,epochs=700,batch_size=1,verbose=1) # verbose=1, 메세지를 출력


# In[9]:


# 시각화
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[10]:


# 예측
model.predict(x_train)


# In[11]:


# 평가
x_test = boston_test[:,:-1]
y_test = boston_test[:,[-1]]
model.evaluate(x_test,y_test)


# In[ ]:




