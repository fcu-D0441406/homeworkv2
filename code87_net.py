import tensorflow as tf
import numpy as np

r = 16
bk_num = [3,2,1,1]
split_block = 6
sp_block_dim = 64

class Resnext:
    
    def __init__(self,img_size,channel,class_num=10,trainable=True):
        self.weight_init = tf.initializers.truncated_normal(0.0,0.01)
        self.weight_decay = tf.contrib.layers.l2_regularizer(0.0001)
        self.x = tf.placeholder(tf.float32,[None,img_size,img_size,channel])
        self.class_num = class_num
        self.build_net(trainable)
        #self.code_87_predict(trainable)
        
    def build_net(self,trainable):
        with tf.variable_scope('resnext',reuse=tf.AUTO_REUSE):
            resnext1 = tf.layers.conv2d(self.x,64,5,2,padding='SAME',
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            resnext1 = tf.layers.batch_normalization(resnext1,training=trainable)
            resnext1 = tf.nn.relu(resnext1)
            #resnext1 = tf.layers.max_pooling2d(resnext1,3,2,padding='SAME')
            print(resnext1)
            self.resnext1 = self.res_block(resnext1,128,bk_num[0],True,trainable)
            print(self.resnext1)
            self.resnext2 = self.res_block(self.resnext1,256,bk_num[1],True,trainable)
            print(self.resnext2)
            
            resnext2_1 = self.batch_relu(tf.layers.conv2d(self.resnext2,128,1,1,padding='SAME'),trainable)
            resnext2_2 = self.batch_relu(tf.layers.conv2d(self.resnext2,128,3,1,padding='SAME'),trainable)
            resnext2_3 = self.batch_relu(tf.layers.conv2d(self.resnext2,128,5,1,padding='SAME'),trainable)
            resnext2_ct = self.Concatenation([resnext2_1,resnext2_2,resnext2_3])
            resnext2_ct = self.batch_relu(tf.layers.conv2d(resnext2_ct,256,1,1,padding='SAME'),trainable)
            resnext2_ct = self.batch_relu(tf.layers.conv2d_transpose(resnext2_ct,256,3,2,padding='SAME'),trainable)
            
            resnext1 = tf.layers.conv2d(self.resnext1,256,1,1,padding='SAME')
            print(resnext1)
            self.code_87_f = self.Concatenation([resnext1,resnext2_ct])
            self.code_87_f = self.batch_relu(tf.layers.conv2d(self.code_87_f,512,3,1,padding='SAME'),trainable)
            self.code_87_pre = tf.layers.conv2d_transpose(self.code_87_f,3,4,4,padding='SAME')
            print(self.code_87_pre)
            
            #self.code_87_pre = tf.layers.conv2d(self.code_87_f,2,3,1,padding='SAME')
            self.code87_softmax = tf.nn.softmax(self.code_87_pre)
            '''
            self.fpn4 = self.fpn_net(self.resnext4,None,True,trainable)
            self.fpn3 = self.fpn_net(self.fpn4,self.resnext3,False,trainable)
            self.fpn2 = self.fpn_net(self.fpn3,self.resnext2,False,trainable)
            self.fpn1 = self.fpn_net(self.fpn2,self.resnext1,False,trainable)
            '''
    
    def batch_relu(self,x,trainable):
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        return x
           
    def res_block(self,x,out_dim,block_num,flag,trainable):
        
        for i in range(block_num):
            if(flag==False):
                stride = 1
            else:
                stride = 2
            res_block = self.merge_block(x,stride,trainable)
            res_block = self.transition_layer(res_block,out_dim,trainable)
            
            if(flag):
                #pre_block = tf.layers.average_pooling2d(x,2,2,padding='SAME')
                #pre_block = tf.pad(pre_block,[[0,0],[0,0],[0,0],[channel,channel]])
                pre_block = tf.layers.conv2d(x,out_dim,3,2,padding='SAME')
                flag = False
            else:
                pre_block = x
            #print(pre_block,res_block)
            #print(pre_block,res_block)
            x = pre_block+res_block
        return x
            
   
    def transition_layer(self,x,out_dim,trainable):
        x = tf.layers.conv2d(x,out_dim,1,1)
        x = tf.layers.batch_normalization(x,training=trainable)
        #print(out_dim)
        return x
    
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
        
    
    def merge_block(self,x,stride,trainable): 
        sp_block = list()
        for i in range(split_block):
            sp_block.append(self.bottleneck(x,stride,trainable))
        return self.Concatenation(sp_block)
            
    def bottleneck(self,x,stride,trainable):
        #x = tf.contrib.layers.batch_norm(x,is_training=trainable)
        if(stride!=1):
            x = self.attention_layer(x)
        else:
            x = tf.layers.conv2d(x,sp_block_dim,1,stride,padding='SAME',
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            x = tf.layers.batch_normalization(x,training=trainable)
            x = tf.nn.relu(x)
       

        #x = tf.contrib.layers.batch_norm(x,is_training=trainable)
        x = tf.layers.conv2d(x,sp_block_dim,3,1,padding='SAME',
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        
        return x
    
    def Concatenation(self,layers) :
        return tf.concat(layers, axis=3)
    
    def fpn_net(self,pre_net,now_net,first,trainable):
        if(first==True):
            #net = tf.layers.conv2d(pre_net,256,3,1,padding='SAME',activation=tf.nn.relu)
            net = tf.layers.conv2d(pre_net,256,3,1,padding='SAME',
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            net = tf.layers.batch_normalization(net,training=trainable)
            net = tf.nn.relu(net)
        else:
            pre = tf.layers.conv2d_transpose(pre_net,256,3,2,padding='SAME')
            #pre = tf.image.resize_images(pre_net,(2,2),0)
            now = tf.layers.conv2d(now_net,256,1,1,padding='SAME',
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            now = tf.layers.batch_normalization(now,training=trainable)
            now = tf.nn.relu(now)
            net = tf.add(pre,now)
            
        return net
    
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