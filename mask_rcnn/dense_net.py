import tensorflow as tf
import numpy as np
import os
class Dense_net:
    
    def __init__(self,img_size,channel,class_num,k,grow_rate,trainable=True):
        self.img_size = img_size
        self.channel = channel
        self.class_num = class_num
        self.drop_rate = 0.25
        self.k = k
        self.grow_rate = grow_rate
        self.x = tf.placeholder(tf.float32,[None,self.img_size,self.img_size,self.channel])
        self.mrsa_init = tf.contrib.layers.variance_scaling_initializer()
        self.build_net(trainable)
        #self.upsample(trainable)
    
    def build_net(self,trainable):
        with tf.variable_scope('dense_net'):
            dense1 = tf.layers.conv2d(self.x,2*self.k,7,2,padding='SAME')
            dense1 = tf.layers.batch_normalization(tf.nn.relu(dense1),training=trainable)
            self.dense1 = tf.layers.max_pooling2d(dense1,3,2,padding='SAME')
            print(self.dense1)
            self.dense2 = self.dense_block(self.dense1,6,trainable)
            print(self.dense2)
            
            self.dense3 = self.dense_block(self.dense2,12,trainable)
            print(self.dense3)
            
            self.dense4 = self.dense_block(self.dense3,24,trainable)
            print(self.dense4)
            '''
            dense5 = self.dense_block(dense4,16,trainable)
            print(dense5)
            '''
            
            avg_pool = tf.layers.average_pooling2d(self.dense4,2,1)
            print(avg_pool)
            flat = tf.layers.flatten(avg_pool)
            print(flat)
            self.prediction = tf.layers.dense(flat,self.class_num)
            print(self.prediction)
    
    def FPN_net(self):
        ch = self.dense4.shape[3]
        self.pre1 = tf.add(tf.layers.conv2d_transpose(self.dense4,ch,3,2,padding='SAME'),
                           tf.layers.conv2d(self.dense3,ch.shape[3],1,1))
        self.pre1 = tf.layers.conv2d(self.pre1,ch,3,1,padding='SAME')
        self.pre2 = tf.add(tf.layers.conv2d_transpose(self.pre1,ch,3,2,padding='SAME'),
                           tf.layers.conv2d(self.dense2,ch,1,1))
        self.pre2 = tf.layers.conv2d(self.pre2,ch,3,1,padding='SAME')
        self.pre3 = tf.add(tf.layers.conv2d_transpose(self.pre2,ch,3,2,padding='SAME'),
                           tf.layers.conv2d(self.dense1,ch,1,1))
        self.pre2 = tf.layers.conv2d(self.pre3,ch,3,1,padding='SAME')
    
    def dense_block(self,x,block_num,trainable):
        for i in range(block_num):
            if(i==0):
                x = self.bottleneck(x,trainable)
                #print(x)
            else:
                con_x = self.bottleneck(x,trainable)
                x = tf.concat([x,con_x],axis=3)
                #print(x)
        ch = (self.k*block_num)*self.grow_rate
        #print(ch)
        
        x = tf.layers.conv2d(x,ch,1,1,padding='SAME')
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x,self.drop_rate)
        x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
        return x
    
    def bottleneck(self,x,trainable):
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,self.k*4,1,1,padding='SAME')
        x = tf.layers.dropout(x,self.drop_rate)
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,self.k,3,1,padding='SAME')
        x = tf.layers.dropout(x,self.drop_rate)
        return x
    
    def upsample(self,trainable):
        self.deconv1 = self.up_dense_block(self.dense3,self.dense4,24,trainable)
        print(self.deconv1)
        self.deconv2 = self.up_dense_block(self.dense2,self.deconv1,12,trainable)
        print(self.deconv2)
        self.deconv3 = self.up_dense_block(self.dense1,self.deconv2,6,trainable)
        print(self.deconv3)
        self.deconv4 = tf.layers.batch_normalization(self.deconv3,training=trainable)
        self.deconv4 = tf.nn.relu(self.deconv4)
        self.deconv4 = tf.layers.conv2d_transpose(self.deconv4,self.class_num,5,4,padding='SAME')
        print(self.deconv4)
    
    def up_dense_block(self,pre_net,x,block_num,trainable):
        for i in range(block_num):
            if(i==0):
                x = self.bottleneck(x,trainable)
                #print(x)
            else:
                con_x = self.bottleneck(x,trainable)
                x = tf.concat([x,con_x],axis=3)
                #print(x)
        ch = int((self.k*block_num)*self.grow_rate)
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        
        x = tf.layers.conv2d_transpose(x,ch,3,2,padding='SAME')
        #print(pre_net)
        #print(x)
        x = tf.concat([x,pre_net],axis=3)
        return x

if(__name__=='__main__'):
    ds = Dense_net(64,3,10,12,0.5)
    #print(resnet.predict)
