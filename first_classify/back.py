import tensorflow as tf
import numpy as np
import os

'''
self.net4 = self.block(self.net3,512,2,is_training=is_training,scope='block4')
print(self.net4)#/16
'''

class ResNet():#resnet34(3,4,6,3) resnet18(2,2,2,2)
    def __init__(self,class_num=10,is_training=True,scope='resnet'):
        img_size = 64
        self.class_num = class_num
        with tf.variable_scope(scope):
            self.x = tf.placeholder(tf.float32,[None,img_size,img_size,3])/255.0
            net = tf.layers.conv2d(self.x,64,5,2,padding='SAME')
            net = tf.nn.relu(tf.layers.batch_normalization(net,training=is_training))
            net = tf.layers.max_pooling2d(net,3,2,padding='SAME')
            print(net)
            
            self.net1 = self.block(net,64,2,is_training=is_training,scope='block1')
            print(self.net1)#/2
            
            self.net2 = self.block(self.net1,128,2,is_training=is_training,scope='block2')
            print(self.net2)#/4
            
            self.net3 = self.block(self.net2,256,2,is_training=is_training,scope='block3')
            print(self.net3)#/8
            
            self.pooling = tf.layers.average_pooling2d(self.net3,[4,4],1)
            print(self.pooling)
            net = tf.layers.flatten(self.pooling,'flatten_layer')
            self.prediction = tf.layers.dense(net,class_num)
            #self.prediction = tf.nn.softmax(net)
            print(self.prediction)
            self.unsample()
        
    def FPN_net(self):
        self.pre1 = tf.add(tf.layers.conv2d_transpose(self.net3,256,3,2,padding='SAME'),
                           tf.layers.conv2d(self.net2,256,1,1))
        self.pre1 = tf.layers.conv2d(self.pre1,256,3,1,padding='SAME')
        self.pre2 = tf.add(tf.layers.conv2d_transpose(self.pre1,256,3,2,padding='SAME'),
                           tf.layers.conv2d(self.net1,256,1,1))
        self.pre2 = tf.layers.conv2d(self.pre2,256,3,1,padding='SAME')
    
    
    
    def block(self,x, n_out, n, is_training=True, scope="block"):
        with tf.variable_scope(scope):
            for i in range(n):
                scope_name = scope+'_'+str(i)
                x = self.bottleneck(x,n_out,is_training=is_training,scope=scope_name)
                #print(x)
        return x
    
    def bottleneck(self,x, n_out, is_training=True, scope="bottleneck"):
        if(x.get_shape()[-1]==n_out):
            stride = 1
        else:
            stride = 2
        with tf.variable_scope(scope):
            net = tf.nn.relu(tf.layers.batch_normalization(x,training=is_training))
            net = tf.layers.conv2d(net,n_out,3,stride,padding='SAME')
            net = tf.nn.relu(tf.layers.batch_normalization(net,training=is_training))
            net = tf.layers.conv2d(net,n_out,3,1,padding='SAME')

            if (x.get_shape()[-1]==n_out):
                shortcut = x
            else:
                shortcut = tf.layers.conv2d(x,n_out,1,2,padding='SAME')
            return tf.add(net,shortcut)
    
    def unsample(self,scope='upresnet'):
        img_size = 64
        with tf.variable_scope(scope):
            #self.unconv1 = self.upblcok(self.pooling,256,4,0)
            self.unconv1 = tf.layers.conv2d_transpose(self.pooling,256,4,1)
            print(self.unconv1)
            self.unconv2 = self.upblock(self.unconv1,128,2,scope='upblock1')
            print(self.unconv2)
            self.unconv3 = self.upblock(self.unconv2,64,2,scope='upblock2')
            print(self.unconv3)
            self.unconv4 = self.upblock(self.unconv3,64,2,scope='upblock3')
            print(self.unconv4)
            self.unconv5 = self.upblock(self.unconv4,2,1,scope='upblock4')
            print(self.unconv5)
            
    def upblock(self,x, n_out, n, is_training=True, scope="upblock"):
        with tf.variable_scope(scope):
            for i in range(n):
                scope_name = scope+'_'+str(i)
                x = self.bottleneck(x,n_out,is_training=is_training,scope=scope_name)
                #print(x)
        return x
    
    def upbottleneck(self,x, n_out, is_training=True, scope="bottleneck"):
        if(x.get_shape()[-1]==n_out):
            stride = 1
        else:
            stride = 2
        with tf.variable_scope(scope):
            net = tf.nn.relu(tf.layers.batch_normalization(x,training=is_training))
            net = tf.layers.conv2d(net,n_out,3,stride,padding='SAME')
            net = tf.nn.relu(tf.layers.batch_normalization(net,training=is_training))
            net = tf.layers.conv2d(net,n_out,3,1,padding='SAME')

            if (x.get_shape()[-1]==n_out):
                shortcut = x
            else:
                shortcut = tf.layers.conv2d_transpose(x,n_out,1,2,padding='SAME')
            return tf.add(net,shortcut)
            
    
    
    
    
class ResNet_50():
    def __init__(self,class_num=10,is_training=True,scope='resnet'):
        img_size = 64
        with tf.variable_scope(scope):
            self.x = tf.placeholder(tf.float32,[None,img_size,img_size,3])
            net = tf.layers.conv2d(self.x,64,3,2,padding='SAME')
            net = tf.nn.relu(tf.layers.batch_normalization(net,training=is_training))
            net = tf.layers.max_pooling2d(net,3,2,padding='SAME')
            print(net)
            net = self.block(net,64,2,is_training=is_training,scope='block1')
            print(net)
            net = self.block(net,128,2,is_training=is_training,scope='block2')
            print(net)
            net = self.block(net,256,2,is_training=is_training,scope='block3')
            print(net)
            '''
            net = self.block(net,512,3,is_training=is_training,scope='block4')
            print(net)
            '''
            net = tf.layers.average_pooling2d(net,[4,4],1)
            net = tf.layers.flatten(net,'flatten_layer')
            self.prediction = tf.layers.dense(net,class_num)
            print(self.prediction)
            #self.prediction = tf.nn.softmax(net)
            
    def block(self,x, n_out, n, is_training=True, scope="block"):
        with tf.variable_scope(scope):
            for i in range(n):
                scope_name = scope+'_'+str(i)
                x = self.bottleneck(x,n_out,i,is_training=is_training,scope=scope_name)
                    
        return x
    
    def bottleneck(self,x, n_out,time,is_training=True, scope="bottleneck"):
        if(time==0 and x.get_shape()[-1]!=n_out):
            stride = 2
        else:
            stride = 1
        with tf.variable_scope(scope):
            net = tf.nn.relu(tf.layers.batch_normalization(x,training=is_training))
            net = tf.layers.conv2d(net,n_out,1,stride,padding='SAME')
            net = tf.nn.relu(tf.layers.batch_normalization(net,training=is_training))
            net = tf.layers.conv2d(net,n_out,3,1,padding='SAME')
            net = tf.nn.relu(tf.layers.batch_normalization(net,training=is_training))
            net = tf.layers.conv2d(net,n_out*4,1,1,padding='SAME')

            if(x.get_shape()[-1]==n_out):
                shortcut = tf.layers.conv2d(x,n_out*4,1,1,padding='SAME')
            elif(time==0):
                shortcut = tf.layers.conv2d(x,n_out*4,1,2,padding='SAME')
            else:
                shortcut = tf.layers.conv2d(x,n_out*4,1,1,padding='SAME')
            return tf.add(net,shortcut)
'''
if(__name__=='__main__'):
    x = tf.random_normal([32, 64, 64, 1])
    resnet = ResNet_50(x)
    print(resnet.prediction)
'''

