#!/usr/bin/env python
# coding: utf-8

# In[1]:


# mnist_cnn
# MNIST and Convolutional Neural Network
# L1,L2 : conv2d + relu + max_pool 
# L3 : FC(Fully Connected Layer)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(5)


# In[2]:


# mnist 데이터 가져오기
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)


# In[3]:


# one-hot 인코딩
nb_classes = 10  # 분류 class의 갯수(0~9)

Y_one_hot = tf.one_hot(y_train,nb_classes)   # (60000, 10)
print(Y_one_hot.shape)                       # (60000, 10) , (2차원)

# X값의 타입을 float형으로 변환
x_train = tf.cast(x_train,dtype=tf.float32)
print(x_train.shape,x_train.dtype)

x_test = tf.cast(x_test,dtype=tf.float32)
print(x_test.shape,x_test.dtype)


# In[4]:


# X값의 shape을 4차원으로 변환
x_img = tf.reshape(x_train,[-1,28,28,1])
print(x_img.shape) # (60000, 28, 28, 1)


# In[5]:


# Layer 1 : conv2d - relu - max_pool
# (?,28,28,1) --> (?, 14, 14, 32)

# <1> conv2d
# L1 input image shape : (?, 28, 28, 1)
# filter : (3,3,1,32), 필터 32개
# strides : (1,1,1,1), padding='SAME'
# 출력 이미지 : (28+2 - 3)/1 + 1 = 28
# (?, 28, 28, 1) --> (?, 28, 28, 32)
W1 = tf.Variable(tf.random.normal([3,3,1,32]),name='weight1')

def L1_conv2d(X):
    return tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')

#  <2> relu
def L1_relu(X):
    return tf.nn.relu(L1_conv2d(X)) # shape 변화가 없음

# <3> max_pool
# input image : (?, 28, 28, 32)
# ksize : (1,2,2,1), strides : (1,2,2,1), padding='SAME'
# 출력 이미지 : (28+1 - 2)/2 + 1 = 14
#  (?, 28, 28, 32) -->  (?, 14, 14, 32)
def L1_MaxPool(X):
    return tf.nn.max_pool(L1_relu(X),ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# In[6]:


# Layer 2 : conv2d - relu - max_pool - flatten
# (?, 14, 14, 32) --> (?, 7, 7, 64) -->(?,7*7*64)

# <1> conv2d
# L2 input image shape : (?, 14, 14, 32)
# filter : (3,3,32,64), 필터 64개
# strides : (1,1,1,1), padding='SAME'
# 출력 이미지 : (14+2 - 3)/1 + 1 = 14
# (?, 14, 14, 32) --> (?, 14, 14, 64)
W2 = tf.Variable(tf.random.normal([3,3,32,64]),name='weight2')

def L2_conv2d(X):
    return tf.nn.conv2d(L1_MaxPool(X),W2,strides=[1,1,1,1],padding='SAME')

#  <2> relu
def L2_relu(X):
    return tf.nn.relu(L2_conv2d(X)) # shape 변화가 없음

# <3> max_pool
# input image : (?, 14,14,64)
# ksize : (1,2,2,1), strides : (1,2,2,1), padding='SAME'
# 출력 이미지 : (14+1 - 2)/2 + 1 = 7
#  (?, 14,14,64) -->  (?,7,7,64)
def L2_MaxPool(X):
    return tf.nn.max_pool(L2_relu(X),ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# <4> flatten layer : : 다차원 배열을 2차원으로 변환하여 FC layer에 전달한다
def L2_Flat(X):
    return tf.reshape(L2_MaxPool(X),[-1,7*7*64])


# In[7]:


# Layer 3 : FC(Fully Connected Layer), 출력층
# (?,7*7*64) * (7*7*64,10) = (?, 10)
W3 = tf.Variable(tf.random.normal([7*7*64,nb_classes]), name='weight3')
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')


# In[8]:


# 예측 함수(hypothesis) : H(X) = softmax(W*X + b)
def logits(X):
    return tf.matmul(L2_Flat(X),W3) + b

def hypothesis(X):
    return tf.nn.softmax(logits(X)) 


# In[9]:


# 방법 2. batch 사이즈로 나누어 학습, 효율적 이며 학습 시간 단축
# 학습 시작

training_epoch = 50
batch_size = 600

# 경사 하강법
# learning_rate(학습율)을 0.01 로 설정하여 optimizer객체를 생성
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Y_one_hot = tf.one_hot(y_train,nb_classes)   # (60000, 10)

print('***** Start Learning!!')
for epoch in range(training_epoch): # 15회
    
    avg_cost = 0
    
    # 100 = 60000/600
    total_batch = int(x_train.shape[0]/batch_size)
    for k in range(total_batch):  # 100회
        batch_xs = x_train[0 + k*batch_size:batch_size + k*batch_size]   # 600개의 X 데이터
        batch_ys = Y_one_hot[0 + k*batch_size:batch_size + k*batch_size] # 600개의 Y 데이터
        
        # X값의 shape을 4차원으로 변환
        X_img = tf.reshape(batch_xs,[-1,28,28,1])
        
        # 비용함수        
        def cost_func_batch():
            cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits(X_img),
                                             labels = batch_ys)
            cost =  tf.reduce_mean(cost_i)
            return cost
        
        # cost를 minimize 한다
        optimizer.minimize(cost_func_batch,var_list=[W1,W2,W3,b])
        avg_cost += cost_func_batch().numpy()/total_batch
            
    print('Epoch:','%04d'%(epoch + 1),'cost:','{:.9f}'.format(avg_cost))
             
print('***** Learning Finished!!')


# In[11]:


# 정확도 측정 : accuracy computation

# y_test 값의 one-hot 인코딩
Y_one_hot = tf.one_hot(y_test,nb_classes)    # (10000,10)
print(Y_one_hot.shape)                       # (10000,10)  , (2차원)

# tf.argmax() : 값이 가장 큰 요소의 인덱스 값을 반환
def predict(X):
    return tf.argmax(hypothesis(X),axis=1)

# X값의 shape을 4차원으로 변환
X_img = tf.reshape(x_test,[-1,28,28,1])

correct_predict = tf.equal(predict(X_img),tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype = tf.float32))
print("Accuracy:",accuracy.numpy()) # Accuracy: 0.9534

#예측
print('***** Predict')
pred = predict(X_img).numpy()
print(pred,y_test)


# In[12]:


# 정확도 비교

# [1] softmax 사용
# 1 layers              -------> Accuracy  : 0.8871
# 4 layers  sigmoid     -------> Accuracy  : 0.9033
# 4 layers  relu        -------> Accuracy  : 0.9534  

# [2] CNN 사용
# 3 layers              -------> Accuracy  : 0.9743 (epoch=15) ,GPU
# 3 layers              -------> Accuracy  : 0.9804 (epoch=50) ,GPU


# In[ ]:


# 실습 과제
# mnist_cnn_deep
# MNIST and Convolutional Neural Network
# L1,L2,L3 : conv2d + relu + max_pool 
# L4,L5 : FC(Fully Connected Layer)
# 출력 size : 32(L1) --> 64(L2)-->128(L3) --> 512(L4) --> 10(L5)

