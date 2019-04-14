import tensorflow as tf
import numpy as np
import os

r = 16

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
        self.code_87_predict(trainable)
        #self.upsample(trainable)
    
    def build_net(self,trainable):
        with tf.variable_scope('dense_net'):
            dense1 = tf.layers.conv2d(self.x,2*self.k,7,2,padding='SAME')
            dense1 = tf.layers.batch_normalization(tf.nn.relu(dense1),training=trainable)
            #dense1 = tf.layers.max_pooling2d(dense1,3,2,padding='SAME')
            #print(dense1)
            self.resnext1 = self.dense_block(dense1,12,True,trainable)
            print(self.resnext1)
            
            self.resnext2 = self.dense_block(self.resnext1,4,True,trainable)
            print(self.resnext2)
            #self.resnext2 = self.attention_layer2(self.resnext2,self.resnext2.shape[-1])
            
            self.resnext3 = self.dense_block(self.resnext2,4,True,trainable)
            print(self.resnext3)
            
            self.resnext4 = self.dense_block(self.resnext3,4,True,trainable)
            print(self.resnext4)
            
            self.fpn4 = self.fpn_net(self.resnext4,None,True,trainable)
            self.fpn3 = self.fpn_net(self.fpn4,self.resnext3,False,trainable)
            self.fpn2 = self.fpn_net(self.fpn3,self.resnext2,False,trainable)
            self.fpn1 = self.fpn_net(self.fpn2,self.resnext1,False,trainable)
            
    
    def fpn_net(self,pre_net,now_net,first,trainable):
        if(first==True):
            #net = tf.layers.conv2d(pre_net,256,3,1,padding='SAME',activation=tf.nn.relu)
            net = tf.layers.conv2d(pre_net,256,3,1,padding='SAME',
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            net = tf.layers.batch_normalization(net,training=trainable)
            net = tf.nn.relu(net)
        else:
            pre = tf.layers.conv2d_transpose(pre_net,256,3,2,padding='SAME')
            #now = tf.layers.conv2d(now_net,256,1,1,padding='SAME',activation=tf.nn.relu)
            now = tf.layers.conv2d(now_net,256,1,1,padding='SAME',
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            now = tf.layers.batch_normalization(now,training=trainable)
            now = tf.nn.relu(now)
            #print(pre,now)
            net = tf.add(pre,now)
            
        return net
    
    
    def dense_block(self,x,block_num,flag,trainable):
        layer_concat = list()
        layer_concat.append(x)
        x = self.bottleneck(x,trainable)
        layer_concat.append(x)
        
        for i in range(block_num-1):
            x = self.Concatenation(layer_concat)
            x = self.bottleneck(x,trainable)
            layer_concat.append(x)
            x = self.Concatenation(layer_concat)
                    
        if(flag==True):
            ch = (self.k*block_num)*self.grow_rate
            x = tf.layers.batch_normalization(x,training=trainable)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x,ch,1,1,padding='SAME')
            x = tf.layers.dropout(x,self.drop_rate,training=trainable)
            x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
            #x = self.attention_layer(x)
        return x
    
    def bottleneck(self,x,trainable):
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,self.k*4,1,1,padding='SAME',
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        x = tf.layers.dropout(x,self.drop_rate,training=trainable)
        
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,self.k,3,1,padding='SAME',
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        x = tf.layers.dropout(x,self.drop_rate,training=trainable)
        return x
    
    def Concatenation(self,layers) :
        return tf.concat(layers, axis=3)
    
    def attention_layer(self,x):
        avg_w = x.shape[1]
        avg_h = x.shape[2]
        #print(avg_w)
        ch = x.shape[3]
        at_x = tf.layers.average_pooling2d(x,(avg_w,avg_h),1,padding='VALID')
        at_x = tf.layers.conv2d(at_x,ch//r,1,1,padding='SAME')
        at_x = tf.nn.relu(at_x)
        at_x = tf.layers.conv2d(at_x,ch,1,1,padding='SAME')
        at_x = tf.nn.sigmoid(at_x)
        x = tf.multiply(x,at_x)
        x = tf.layers.average_pooling2d(x,2,2,padding='SAME')
        #print(x)
        return x
    
    def attention_layer2(self,x, ch, sn=False, scope='attention', reuse=False):
        #print(x,ch)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            f = tf.layers.conv2d(x,ch//8,1,1)
            g = tf.layers.conv2d(x,ch//8,1,1)
            h = tf.layers.conv2d(x,ch,1,1)
            
            f = tf.reshape(f,(x.shape[0],-1,f.shape[-1]))
            g = tf.reshape(g,(x.shape[0],-1,g.shape[-1]))
            h = tf.reshape(h,(x.shape[0],-1,h.shape[-1]))
            
            s = tf.matmul(g,f,transpose_b=True)
            beta = tf.nn.softmax(s)
            o = tf.matmul(beta,h)
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
            o = tf.reshape(o,shape=x.shape)
            x = gamma*o+x
            
        return x
    
    def code_87_predict(self,trainable):
        self.code_87_f = tf.layers.conv2d(self.fpn1,512,1,1,padding='SAME')
        #self.code_87_f = tf.layers.conv2d(self.resnext1,512,3,1,padding='SAME')
        self.code_87_f = tf.layers.batch_normalization(self.code_87_f,training=trainable)
        self.code_87_f = tf.nn.relu(self.code_87_f)
        self.code_87_f = tf.layers.conv2d(self.code_87_f,512,1,1,padding='SAME')
        self.code_87_f = tf.layers.batch_normalization(self.code_87_f,training=trainable)
        self.code_87_f = tf.nn.relu(self.code_87_f)
        self.code_87_pre = tf.layers.conv2d(self.code_87_f,2,1,1,padding='SAME')
        self.code87_softmax = tf.nn.softmax(self.code_87_pre)
        print(self.code_87_pre)
    
    

