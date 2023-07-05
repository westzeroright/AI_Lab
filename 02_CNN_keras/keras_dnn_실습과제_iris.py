#!/usr/bin/env python
# coding: utf-8

# In[1]:


# iris_softmax_multi_classification

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(5)

species_list =['"setosa"','"versicolor"','"virginica"']

xy = np.loadtxt('iris.csv',delimiter=',',dtype=np.str,skiprows=1)
xy.shape


# In[2]:


x_train = np.float32(xy[:35,1:-1])
x_train = np.append(x_train , np.float32(xy[50:85,1:-1]),0)
x_train = np.append(x_train , np.float32(xy[100:135,1:-1]),0) # [105,4]

y_train = xy[:35,[-1] ]
y_train = np.append(y_train, xy[50:85,[-1]],0)
y_train = np.append(y_train, xy[100:135,[-1]],0) # [105,1]

for i in range(105):
   y_train[i,-1] = np.int32(species_list.index(y_train[i,-1]))
print(y_train)


# In[3]:


x_test = np.float32(xy[35:50,1:-1])
x_test = np.append(x_test , np.float32(xy[85:100,1:-1]),0)
x_test = np.append(x_test , np.float32(xy[135:,1:-1]),0) # [45,4]

y_test = xy[35:50,[-1] ]
y_test = np.append(y_test, xy[85:100,[-1]],0)
y_test = np.append(y_test, xy[135:,[-1]],0) # [45,1]

for i in range(45):
   y_test[i,-1] = np.int32(species_list.index(y_test[i,-1]))
print(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# X = np.array(x_train,dtype=np.float32)
Y = np.array(y_train,dtype=np.int32)   # 반드시 int형으로(one_hot encoding)


# In[4]:


# [one-hot 인코딩]
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)


# In[5]:


# Dense Layer 구현 : 2층
model = tf.keras.Sequential([
    # 첫번째 층 출력 : [None,20],   활성화 함수 : 'relu', metrics:['accuracy']
    tf.keras.layers.Dense(units=20,activation='relu',input_shape=(4,)) ,
    # 두번째 층 출력 : [Non,2],   활성화 함수 : 'relu'
    tf.keras.layers.Dense(units=3,activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.summary()


# In[6]:


# 학습
history = model.fit(x_train,y_train,epochs=700,batch_size=1,verbose=1) # verbose=1, 메세지를 출력


# In[7]:


# 시각화
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[8]:


# 예측
preds = model.predict(x_train)
np.round(preds)


# In[10]:


# 평가
model.evaluate(x_test,y_test)


# In[11]:


# Dense Layer 구현 : 3층
model = tf.keras.Sequential([
    # 첫번째 층 출력 : [None,20],   활성화 함수 : 'relu', metrics:['accuracy']
    tf.keras.layers.Dense(units=40,activation='relu',input_shape=(4,)) ,
    # 첫번째 층 출력 : [None,20],   활성화 함수 : 'relu', metrics:['accuracy']
    tf.keras.layers.Dense(units=20,activation='relu',input_shape=(4,)) ,
    # 두번째 층 출력 : [Non,2],   활성화 함수 : 'relu'
    tf.keras.layers.Dense(units=3,activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.summary()


# In[12]:


# 학습
history = model.fit(x_train,y_train,epochs=700,batch_size=1,verbose=1) # verbose=1, 메세지를 출력


# In[13]:


# 시각화
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[14]:


# 예측
preds = model.predict(x_train)
np.round(preds)


# In[15]:


# 평가
model.evaluate(x_test,y_test)


# In[ ]:




