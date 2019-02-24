import tensorflow as tf
import numpy as np
import os
import cv2

class Dense_net:
    
    def __init__(self,img_size,channel,class_num,k,grow_rate,trainable=True):
        self.img_size = img_size
        self.channel = channel
        self.class_num = class_num
        self.drop_rate = 0.3
        self.k = k
        self.grow_rate = grow_rate
        self.x = tf.placeholder(tf.float32,[None,self.img_size,self.img_size,self.channel])
        self.x_loc = tf.placeholder(tf.float32,[None,4])
        
        self.ratio = [0.5,1,2]
        self.anchor_num = 9
        self.iou_thresh = 0.6
        self.build_net(trainable)
        #self.upsample(trainable)
    
    def build_net(self,trainable):
        with tf.variable_scope('dense_net'):
            #tf.contrib.layers.variance_scaling_initializer()
            dense1 = tf.layers.conv2d(self.x,2*self.k,7,1,padding='SAME',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            dense1 = tf.layers.max_pooling2d(dense1,3,2,padding='SAME')
            self.dense1 = self.dense_block(dense1,6,trainable)
            print(self.dense1)
            self.dense2 = self.dense_block(self.dense1,6,trainable)
            print(self.dense2)
            
            self.dense3 = self.dense_block(self.dense2,12,trainable)
            print(self.dense3)
            
            self.dense4 = self.dense_block(self.dense3,16,trainable)
            print(self.dense4)
            
            '''
            dense5 = self.dense_block(dense4,16,trainable)
            print(dense5)
            '''
            
            avg_pool = tf.layers.average_pooling2d(self.dense3,2,1)
            #print(avg_pool)
            flat = tf.layers.flatten(avg_pool)
            #print(flat)
            self.prediction = tf.layers.dense(flat,self.class_num)
            #print(self.prediction)
            
    
    def fpn_net(self):
        ch = self.dense4.shape[3]
        
        self.pre0 = tf.layers.conv2d(self.dense4,ch,1,1,padding='SAME')
        #print(self.pre0)
        
        self.pre1 = tf.add(tf.layers.conv2d_transpose(self.pre0,ch,3,2,padding='SAME'),
                           tf.layers.conv2d(self.dense3,ch,1,1,padding='SAME'))
        self.pre1 = tf.layers.conv2d(self.pre1,ch,3,1,padding='SAME')
        #print(self.pre1)
        self.pre2 = tf.add(tf.layers.conv2d_transpose(self.pre1,ch,3,2,padding='SAME'),
                           tf.layers.conv2d(self.dense2,ch,1,1,padding='SAME'))
        self.pre2 = tf.layers.conv2d(self.pre2,ch,3,1,padding='SAME')
        #print(self.pre2)
        self.pre3 = tf.add(tf.layers.conv2d_transpose(self.pre2,ch,3,2,padding='SAME'),
                           tf.layers.conv2d(self.dense1,ch,1,1,padding='SAME'))
        self.pre3 = tf.layers.conv2d(self.pre3,ch,3,1,padding='SAME')
        #print(self.pre3)
    
    def rpn_net(self):
        
        self.r_net0 = tf.layers.conv2d(self.pre0,512,3,1,padding='SAME')
        #print(self.r_net0)
        self.bf_check0 = tf.layers.conv2d(self.r_net0,2*self.anchor_num,1,1,padding='SAME')
        self.loc_check0 = tf.layers.conv2d(self.r_net0,4*self.anchor_num,1,1,padding='SAME')
        self.loc_check0 = tf.reshape(self.loc_check0,(-1,
                                                      self.loc_check0.shape[1],
                                                      self.loc_check0.shape[2],
                                                      self.anchor_num,4))
        print(self.bf_check0)
        print(self.loc_check0)
        
        self.r_net1 = tf.layers.conv2d(self.pre1,512,3,1,padding='SAME')
        #print(self.r_net1)
        self.bf_check1 = tf.layers.conv2d(self.r_net1,2*self.anchor_num,1,1,padding='SAME')
        self.loc_check1 = tf.layers.conv2d(self.r_net1,4*self.anchor_num,1,1,padding='SAME')
        self.loc_check1 = tf.reshape(self.loc_check1,(-1,
                                                      self.loc_check1.shape[1],
                                                      self.loc_check1.shape[2],
                                                      self.anchor_num,4))
        print(self.bf_check1)
        print(self.loc_check1)
        
        self.r_net2 = tf.layers.conv2d(self.pre2,512,3,1,padding='SAME')
        #print(self.r_net2)
        self.bf_check2 = tf.layers.conv2d(self.r_net2,2*self.anchor_num,1,1,padding='SAME')
        self.loc_check2 = tf.layers.conv2d(self.r_net2,4*self.anchor_num,1,1,padding='SAME')
        self.loc_check2 = tf.reshape(self.loc_check2,(-1,
                                                      self.loc_check2.shape[1],
                                                      self.loc_check2.shape[2],
                                                      self.anchor_num,4))
        print(self.bf_check2)    
        print(self.loc_check2)
        
        self.r_net3 = tf.layers.conv2d(self.pre3,512,3,1,padding='SAME')
        #print(self.r_net3)
        self.bf_check3 = tf.layers.conv2d(self.r_net3,2*self.anchor_num,1,1,padding='SAME')
        
        self.loc_check3 = tf.layers.conv2d(self.r_net3,4*self.anchor_num,1,1,padding='SAME')
        self.loc_check3 = tf.reshape(self.loc_check3,(-1,
                                                      self.loc_check3.shape[1],
                                                      self.loc_check3.shape[2],
                                                      self.anchor_num,4))
        print(self.bf_check3) 
        print(self.loc_check3)
    
    def score_softmax(self):
        init_shape = self.bf_check0.shape[2]
        self.bf_check0 = tf.reshape(self.bf_check0,(-1,
                                                    self.bf_check0.shape[1],
                                                    self.bf_check0.shape[2]*self.anchor_num,
                                                    2))
        #print(self.bf_check0)
        
        self.bf_check0 = tf.reshape(self.bf_check0,(-1,
                                                    self.bf_check0.shape[1],
                                                    init_shape,
                                                    self.anchor_num,2))
        self.sbf_check0 = tf.nn.softmax(self.bf_check0)
        #print(self.bf_check0)
        
        init_shape = self.bf_check1.shape[2]
        self.bf_check1 = tf.reshape(self.bf_check1,(-1,
                                                    self.bf_check1.shape[1],
                                                    self.bf_check1.shape[2]*self.anchor_num,
                                                    2))
        
        self.bf_check1 = tf.reshape(self.bf_check1,(-1,
                                                    self.bf_check1.shape[1],
                                                    init_shape,
                                                    self.anchor_num,2))
        self.sbf_check1 = tf.nn.softmax(self.bf_check1)
        
        init_shape = self.bf_check2.shape[2]
        self.bf_check2 = tf.reshape(self.bf_check2,(-1,
                                                    self.bf_check2.shape[1],
                                                    self.bf_check2.shape[2]*self.anchor_num,
                                                    2))
        
        
        self.bf_check2 = tf.reshape(self.bf_check2,(-1,
                                                    self.bf_check2.shape[1],
                                                    init_shape,
                                                    self.anchor_num,2))
        self.sbf_check2 = tf.nn.softmax(self.bf_check2)
        
        init_shape = self.bf_check3.shape[2]
        self.bf_check3 = tf.reshape(self.bf_check3,(-1,
                                                    self.bf_check3.shape[1],
                                                    self.bf_check3.shape[2]*self.anchor_num,
                                                    2))
       
        
        self.bf_check3 = tf.reshape(self.bf_check3,(-1,
                                                    self.bf_check3.shape[1],
                                                    init_shape,
                                                    self.anchor_num,2))
        self.sbf_check3 = tf.nn.softmax(self.bf_check3)
    
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
        x = tf.layers.conv2d(x,self.k*4,1,1,padding='SAME',kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        x = tf.layers.dropout(x,self.drop_rate)
        x = tf.layers.batch_normalization(x,training=trainable)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x,self.k,3,1,padding='SAME',kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
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



img = cv2.imread('test.jpg')
anchor_box = {'0':[96,48,24],'1':[96,48,24],'2':[96,48,24],'3':[96,48,24]}
a_num = 3
anchor_num = 9
ratio = [[1,1],[1,2],[2,1]]
ratio_num = 3

def cal_nms(pre_loc,label_loc):
    
    x1,y1,x2,y2 = pre_loc
    pre_area = (x2-x1)*(y2-y1)
    nms = 0
    result = -1
    for i in range(label_loc.shape[0]):
        l_x1,l_y1,l_x2,l_y2 = label_loc[i][:]
        #print(l_x1,l_y1,l_x2,l_y2)
        label_area = (l_x2-l_x1)*(l_y2-l_y1)
        xx1 = max(x1,l_x1)
        yy1 = max(y1,l_y1)
        xx2 = min(x2,l_x2)
        yy2 = min(y2,l_y2)
        n_width = xx2-xx1
        n_height = yy2-yy1
        if(n_width<0 or n_height<0):
            pass
        else:
            n_area = n_width*n_height
            nms = (n_area*1.0)/(label_area+pre_area-n_area)
            if(nms>=0.7):
                if(nms>=0.8):
                    cv2.rectangle(img, (int(xx1), int(yy1)), (int(xx2), int(yy2)), (0, 255, 0), 2)
                    show_img(img)
                return 1,nms,i
            elif(nms>=0.3):
                #cv2.rectangle(img, (int(xx1), int(yy1)), (int(xx2), int(yy2)), (0, 255, 0), 2)
                #show_img(img)
                result = 0
    return result,nms,-1

def cal_fg_bg_anchor(bf_check,loc,p,label_loc):
    
    r_fg = np.zeros(shape=bf_check.shape)
    r_box = np.zeros([bf_check.shape[0],bf_check.shape[1],anchor_num,label_loc.shape[0]])
    
    record_max_nms = 0
    record_fg = [0,0,0]
    record_box = [0,0,0,0]
    
    positive = False
    for i in range(bf_check.shape[0]):
        for j in range(bf_check.shape[1]):
            for a in range(ratio_num):
                for aa in range(a_num):
                    #if(bf_check[i][j][a][0]>=0.0):
                    #print('hi')
                    height = anchor_box[p][aa]*ratio[a][0]
                    width = anchor_box[p][aa]*ratio[a][1]
                    c_x = (j+1)*anchor_box[p][aa]*ratio[a][1]+width*loc[i][j][a][0]
                    c_y = (i+1)*anchor_box[p][aa]*ratio[a][0]+height*loc[i][j][a][1]
                    #print(loc[i][j][a*4+2])
                    c_width = width*float(np.exp(loc[i][j][a*ratio_num+aa][2]))
                    c_height = height*float(np.exp(loc[i][j][a*ratio_num+aa][3]))
                    
                    x1 = c_x-(c_width/2)
                    y1 = c_y-(c_height/2)
                    x2 = x1+c_width
                    y2 = y1+c_height
    
                    if(x1<0 or y1<0 or x2>224 or y2>224):
                        continue
                    else:
                        pre_loc = [x1,y1,x2,y2]
                        result,n,b = cal_nms(pre_loc,label_loc)
                        if(positive):
                            if(result==1):
                                r_box[i][j][a*ratio_num+aa][b] = 1
                                r_fg[i][j][a*ratio_num+aa][0] = 1
                                r_fg[i][j][a*ratio_num+aa][1] = 0
                            elif(result==-1):
                                r_fg[i][j][a*ratio_num+aa][0] = 0
                                r_fg[i][j][a*ratio_num+aa][1] = 1
                        else:
                            if(n>record_max_nms):
                                record_max_nms  = n
                                record_fg = [i,j,a*ratio_num+aa]
                                record_box = [i,j,a*ratio_num+aa,b]
                            if(result==1):
                                r_box[i][j][a*ratio_num+aa][b] = 1
                                r_fg[i][j][a*ratio_num+aa][0] = 1
                                r_fg[i][j][a*ratio_num+aa][1] = 0
                                positive = True
                            elif(result==-1):
                                r_fg[i][j][a*ratio_num+aa][0] = 0
                                r_fg[i][j][a*ratio_num+aa][1] = 1
    if(positive==False):
        if(record_max_nms>0.5):
            i,j,a = record_fg[:]
            r_fg[i][j][a][0] = 1 
            r_fg[i][j][a][1] = 0 
            i,j,a,b = record_box[:]
            r_box[i][j][a][b] = 1
            #print(record_max_nms,r_fg[i][j][a])
                
    return r_fg,r_box

def rpn_fg_loss(fg,r_fg,fg_batch,p):
    test_array = np.zeros([fg.shape[0],fg.shape[1],anchor_num])
    pos = 0
    neg = 0
    for i in range(r_fg.shape[0]):
        for j in range(r_fg.shape[1]):
            for a in range(3):
                for aa in range(a_num):
                    if(r_fg[i][j][a*ratio_num+aa][0]==1):
                        fg_batch+=1
                        test_array[i][j][a*ratio_num+aa] = 1
                        pos+=1
                    elif(r_fg[i][j][a*ratio_num+aa][1]==1):
                        if(np.random.uniform()<p):
                            fg_batch+=1
                            test_array[i][j][a*ratio_num+aa] = 1
                            neg+=1
    #return np.array(predict_fg),np.array(label_fg)
    print(pos,neg)
    return fg_batch,test_array

def box_regression(r_fg,p_box,r_box,l_box,bx_batch,p):
    ground_box = np.zeros(shape=p_box.shape)
    
    for i in range(r_fg.shape[0]):
        for j in range(r_fg.shape[1]):
            for a in range(3):
                for aa in range(a_num):
                    if(r_fg[i][j][a*ratio_num+aa][0]==1):
                        for b in range(r_box.shape[3]):
                            if(r_box[i][j][a*ratio_num+aa][b]==1):
                                l_x,l_y,l_w,l_h = (l_box[b][0]+l_box[b][2])/2,(l_box[b][1]+l_box[b][3])/2,l_box[b][2]-l_box[b][0],l_box[b][3]-l_box[b][1]                                              
                                p_x,p_y,p_w,p_h = j*anchor_box[p][aa]*ratio[a][1],i*anchor_box[p][aa]*ratio[a][0],anchor_box[p][aa]*ratio[a][1],anchor_box[p][aa]*ratio[a][0]
                                                        
                                ground_box[i][j][a][0] = (l_x-p_x)/p_w
                                ground_box[i][j][a][1] = (l_y-p_y)/p_h
                                ground_box[i][j][a][2] = np.log(l_w/p_w)
                                ground_box[i][j][a][3] = np.log(l_h/p_h)
                                bx_batch+=1
    return bx_batch,ground_box

def cal_fg_loss(real_fg0,fg_check0,real_test_fg0,
                real_fg1,fg_check1,real_test_fg1,
                real_fg2,fg_check2,real_test_fg2,
                real_fg3,fg_check3,real_test_fg3,
                fg_batch):
    
    fg_loss0 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(real_fg0,4),logits=fg_check0)
    fg_loss0 = tf.reduce_sum(tf.multiply(fg_loss0,real_test_fg0))
    fg_loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(real_fg1,4),logits=fg_check1)
    fg_loss1 = tf.reduce_sum(tf.multiply(fg_loss1,real_test_fg1))
    fg_loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(real_fg2,4),logits=fg_check2)
    fg_loss2 = tf.reduce_sum(tf.multiply(fg_loss2,real_test_fg2))
    fg_loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(real_fg3,4),logits=fg_check3)
    fg_loss3 = tf.reduce_sum(tf.multiply(fg_loss3,real_test_fg3))
    loss = (fg_loss0+fg_loss1+fg_loss2+fg_loss3)/fg_batch
    
    return loss

def cal_box_loss(pre_loc,label_loc):
    j3 = tf.add(pre_loc,-1*label_loc)
    j4 = tf.cast(tf.equal(label_loc,0),tf.float32)
    j5 = (1-j4)*j3
    j6 = tf.reduce_sum(j5,axis=4)
    j7 = tf.cast(tf.less(j6,1.0),tf.float32)
    j8 = (j7)*(0.5*j6*j6)+(1.0-j7)*(tf.abs(j6)-0.5)
    j9 = tf.reduce_sum(j8)
    return j9


def train_rpn(net,label_loc):
    img = cv2.imread('test.jpg')
    input_x = np.reshape(img,(-1,224,224,3))
    label_loc = np.reshape(np.array([64,83,155,140],dtype=np.float),(-1,4))
    
    x = ds.x
    fg_check0,fg_check1,fg_check2,fg_check3 = ds.bf_check0,ds.bf_check1,ds.bf_check2,ds.bf_check3
    sfg_check0,sfg_check1,sfg_check2,sfg_check3 = ds.sbf_check0,ds.sbf_check1,ds.sbf_check2,ds.sbf_check3
    loc_check0,loc_check1,loc_check2,loc_check3 = ds.loc_check0,ds.loc_check1,ds.loc_check2,ds.loc_check3
    
    fg_batch = tf.placeholder(tf.float32)
    real_fg0 = tf.placeholder(tf.float32,[None,7,7,anchor_num,2])
    real_test_fg0 = tf.placeholder(tf.float32,[None,7,7,anchor_num])
    
    
    real_fg1 = tf.placeholder(tf.float32,[None,14,14,anchor_num,2])
    real_test_fg1 = tf.placeholder(tf.float32,[None,14,14,anchor_num])
    
    real_fg2 = tf.placeholder(tf.float32,[None,28,28,anchor_num,2])
    real_test_fg2 = tf.placeholder(tf.float32,[None,28,28,anchor_num])
    
    real_fg3 = tf.placeholder(tf.float32,[None,56,56,anchor_num,2])
    real_test_fg3 = tf.placeholder(tf.float32,[None,56,56,anchor_num])
    
    
    bounding_box0 = tf.placeholder(tf.float32,[None,7,7,anchor_num,4])
    bounding_box1 = tf.placeholder(tf.float32,[None,14,14,anchor_num,4])
    bounding_box2 = tf.placeholder(tf.float32,[None,28,28,anchor_num,4])
    bounding_box3 = tf.placeholder(tf.float32,[None,56,56,anchor_num,4])
    
    fg_loss = cal_fg_loss(real_fg0,fg_check0,real_test_fg0,
                       real_fg1,fg_check1,real_test_fg1,
                       real_fg2,fg_check2,real_test_fg2,
                       real_fg3,fg_check3,real_test_fg3,
                       fg_batch)
    
    box_batch = tf.placeholder(tf.float32)
    box_loss0 = cal_box_loss(loc_check0,bounding_box0)
    box_loss1 = cal_box_loss(loc_check1,bounding_box1)
    box_loss2 = cal_box_loss(loc_check2,bounding_box2)
    box_loss3 = cal_box_loss(loc_check3,bounding_box3)
    
    box_loss = (box_loss0+box_loss1+box_loss2+box_loss3)/box_batch
    box_loss = tf.clip_by_value(box_loss,1e-1,100)
    total_loss = tf.add(box_loss,fg_loss)
    #train_op = tf.train.AdamOptimizer(1e-4).minimize(box_loss)
    #train_op = tf.train.AdamOptimizer(1e-4).minimize(fg_loss)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            sfg0,sfg1,sfg2,sfg3 = sess.run(
                            [sfg_check0,sfg_check1,sfg_check2,sfg_check3],feed_dict={x:input_x})
            sfg0,sfg1,sfg2,sfg3 = sfg0[0],sfg1[0],sfg2[0],sfg3[0]
            loc0,loc1,loc2,loc3 = sess.run(
                            [loc_check0,loc_check1,loc_check2,loc_check3],feed_dict={x:input_x})
            loc0,loc1,loc2,loc3 = loc0[0],loc1[0],loc2[0],loc3[0]
            r_fg0,r_box0 = cal_fg_bg_anchor(sfg0,loc0,'0',label_loc)
            r_fg1,r_box1 = cal_fg_bg_anchor(sfg1,loc1,'1',label_loc)
            r_fg2,r_box2 = cal_fg_bg_anchor(sfg2,loc2,'2',label_loc)
            r_fg3,r_box3 = cal_fg_bg_anchor(sfg3,loc3,'3',label_loc)
            r_fg0 = r_fg0[np.newaxis,:,:,:,:]
            r_fg1 = r_fg1[np.newaxis,:,:,:,:]
            r_fg2 = r_fg2[np.newaxis,:,:,:,:]
            r_fg3 = r_fg3[np.newaxis,:,:,:,:]

            bs_batch = 0
            bs_batch,t_array0 = rpn_fg_loss(sfg0,r_fg0[0],bs_batch,1.0)
            bs_batch,t_array1 = rpn_fg_loss(sfg1,r_fg1[0],bs_batch,0.8)
            bs_batch,t_array2 = rpn_fg_loss(sfg2,r_fg2[0],bs_batch,0.8)
            bs_batch,t_array3 = rpn_fg_loss(sfg3,r_fg3[0],bs_batch,0.8)
            
            bx_batch = 0
            bx_batch,ground_box0 = box_regression(r_fg0[0],loc0,r_box0,label_loc,bx_batch,'0')
            bx_batch,ground_box1 = box_regression(r_fg1[0],loc1,r_box1,label_loc,bx_batch,'1')
            bx_batch,ground_box2 = box_regression(r_fg2[0],loc2,r_box2,label_loc,bx_batch,'2')
            bx_batch,ground_box3 = box_regression(r_fg3[0],loc3,r_box3,label_loc,bx_batch,'3')
            if(bx_batch==0):
                bx_batch = 1
            
            ls,_ = sess.run([total_loss,train_op],feed_dict={x:input_x,
                                                       real_fg0:r_fg0,real_test_fg0:t_array0[np.newaxis,:,:,:],
                                                       real_fg1:r_fg1,real_test_fg1:t_array1[np.newaxis,:,:,:],
                                                       real_fg2:r_fg2,real_test_fg2:t_array2[np.newaxis,:,:,:],
                                                       real_fg3:r_fg3,real_test_fg3:t_array3[np.newaxis,:,:,:],
                                                       bounding_box0:ground_box0[np.newaxis,:,:,:,:],
                                                       bounding_box1:ground_box1[np.newaxis,:,:,:,:],
                                                       bounding_box2:ground_box2[np.newaxis,:,:,:,:],
                                                       bounding_box3:ground_box3[np.newaxis,:,:,:,:],
                                                       fg_batch:bs_batch,box_batch:bx_batch})
            
            
            print(ls)
            #print(ls,cx[0][0])
            #print(predict_fg.shape,label_fg.shape)
            #rpn_fg_loss(fg3,r_fg3,56)
            
    
def show_img(img):
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if(__name__=='__main__'):
    img = cv2.imread('test.jpg')
    input_x = np.reshape(img,(-1,224,224,3))
    label_loc = np.reshape(np.array([64,83,155,140],dtype=np.float),(-1,4))
    
    ds = Dense_net(224,3,10,12,0.5)
    x = ds.x
    ds.fpn_net()
    ds.rpn_net()
    ds.score_softmax()
    fg_loss = train_rpn(ds,label_loc)

        
