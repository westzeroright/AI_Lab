#!/usr/bin/env python
# coding: utf-8

# In[1]:


# logistic_regression_Caesarian.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(5)

xy = np.loadtxt('caesarian.csv',delimiter=',',dtype=np.float32)

# train data set
x_data = xy[:56, :-1 ]
y_data = xy[:56, [-1] ]

x_train = np.array(x_data,dtype=np.float32)
y_train = np.array(y_data,dtype=np.float32)
x_train.shape,y_train.shape



# In[2]:


# Dense Layer 구현 : 2층
model = tf.keras.Sequential([
    # 첫번째 층 출력 : [None,20],   활성화 함수 : 'relu', metrics:['accuracy']
    tf.keras.layers.Dense(units=20,activation='relu',input_shape=(5,)) ,
    # 두번째 층 출력 : [Non,2],   활성화 함수 : 'relu'
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='binary_crossentropy',
             metrics=['accuracy'])

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
preds = model.predict(x_train)
np.round(preds)


# In[6]:


# 평가
x_data = xy[56:, :-1 ]
y_data = xy[56:, [-1] ]

x_test = np.array(x_data,dtype=np.float32)
y_test = np.array(y_data,dtype=np.float32)

model.evaluate(x_test,y_test)


# In[7]:


# Dense Layer 구현 : 3층
model = tf.keras.Sequential([
    # 첫번째 층 출력 : [None,20],   활성화 함수 : 'relu', metrics:['accuracy']
    tf.keras.layers.Dense(units=20,activation='relu',input_shape=(5,)) ,
    # 두번째 층 출력 : [Non,2],   활성화 함수 : 'relu'
    tf.keras.layers.Dense(units=2,activation='relu'),
    # 세번째 층 출력 : [Non,2],   활성화 함수 : 'relu'
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='binary_crossentropy',
             metrics=['accuracy'])

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
preds = model.predict(x_train)
np.round(preds)


# In[11]:


# 평가
x_data = xy[56:, :-1 ]
y_data = xy[56:, [-1] ]

x_test = np.array(x_data,dtype=np.float32)
y_test = np.array(y_data,dtype=np.float32)

model.evaluate(x_test,y_test)


# In[ ]:




