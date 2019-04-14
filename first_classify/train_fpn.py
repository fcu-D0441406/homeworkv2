import tensorflow as tf
#import resnet_v2
import cv2
import numpy as np
import dense_net
import at_resnext
import matplotlib.pyplot as plt
import code87_net

img_size = 256
class_num=2
channel = 3
batch_size = 8
test_size = 1
tfrecord_train = './fpn_train.tfrecords'
tfrecord_test = './fpn_test.tfrecords'

ckpts = './checkpoint0_dir'
adam_meta = './checkpoint0_dir/MyModel'


def read_and_decode(tfrecord_file_path,batch_size):
    tfrecord_file = tf.train.string_input_producer([tfrecord_file_path])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(tfrecord_file)
    img_features = tf.parse_single_example(serialized_example,features={
                                        'label_raw':tf.FixedLenFeature([],tf.string),
                                        'image_raw':tf.FixedLenFeature([],tf.string),
                                        })
    image = tf.decode_raw(img_features['image_raw'],tf.uint8)
    image = tf.reshape(image,[img_size,img_size,channel])
    label = tf.decode_raw(img_features['label_raw'],tf.uint8)
    label = tf.reshape(label,[64,64,class_num])
    image_batch,label_batch = tf.train.shuffle_batch([image,label],
                                                     batch_size=batch_size,
                                                     min_after_dequeue=5,
                                                     num_threads=4,
                                                     capacity=7)
    return image_batch,label_batch

def show_img(img):
    if(img.shape[-1]==1):
        plt.imshow(img,cmap ='gray')
        plt.show()
    else:
        plt.imshow(img)
        plt.show()

def show_feature(sess,x,img,feature):
    #print(img.shape)
    feature_map = sess.run(feature,feed_dict={x:img})
    show_img(feature_map[:,:,0])

def add_neg_weight(mask):
    mask = np.reshape(mask,(-1,64*64,2))
    neg_weight = np.ones(shape=[mask.shape[0],mask.shape[1]],dtype=np.float32)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if(mask[i][j][0]!=1):
                neg_weight[i][j]*=5.0
    neg_weight = np.reshape(neg_weight,(-1,64,64))
    return neg_weight

def show_train_mask(result,batch_x,batch_y):
    result2 = np.reshape(result,(-1,64,64,2))
    image = batch_x[0]
    mask = batch_y[0]
    show_img(image)
    show_img(mask[:,:,1])
    show_img(result2[:,:,1])

def show_test_mask(result,batch_x):
    result2 = np.reshape(result,(-1,64,64,2))
    mask = np.zeros(shape=[result2.shape[0],64,64],dtype=np.float32)
    for s in range(test_size):
        for i in range(64):
            for j in range(64):
                if(result2[s][i][j][1]>0.5):
                    mask[s][i][j] = result2[s][i][j][1]
    image = batch_x[0]
    mask = mask[0]
    show_img(image)
    show_img(mask)
    
if(__name__=='__main__'):
    
    train_image,train_label = read_and_decode(tfrecord_train,batch_size)
    test_image,test_label = read_and_decode(tfrecord_test,test_size)
    
    y = tf.placeholder(tf.float32,[None,64,64,class_num])
    neg_weight = tf.placeholder(tf.float32,[None,64,64])
    #resnet32 = resnet_v2.ResNet(class_num)
    resnet32 = at_resnext.Resnext(img_size,channel,class_num)
    #resnet32 = code87_net.Resnext(img_size,channel,class_num)
    #resnet32 = dense_net.Dense_net(img_size,channel,class_num,24,0.5)
    x = resnet32.x
    fpn_predict = resnet32.code_87_pre
    #resnet32.upsample(True)
    #resnet32.unsample2()
    
    
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 3), logits=fpn_predict)
    loss = tf.multiply(loss,neg_weight)
    tv = tf.trainable_variables()
    l2_cost = 0.0005* tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
    loss = loss+l2_cost
    #print(loss.shape)
    loss = tf.reduce_mean(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,3),tf.argmax(fpn_predict,3)),'float'))
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        sess.run(tf.global_variables_initializer())
        ########  save
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list,max_to_keep=5)
        ########
        if tf.train.latest_checkpoint(ckpts) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(ckpts))
        else:
            assert 'can not find checkpoint folder path!'
        
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        print('start --------')
        for i in range(2001):
            batch_x,batch_y = sess.run([train_image,train_label])
            neg_w = add_neg_weight(batch_y)
            _,result1,ls,acc = sess.run([train_op,fpn_predict,loss,accuracy],
                                        feed_dict={x:batch_x,y:batch_y,neg_weight:neg_w})
           
            if(i%10==0):
                
                batch_test_x,batch_test_y = sess.run([test_image,test_label])
                result,resnext1,val_acc = sess.run([resnet32.code87_softmax,resnet32.code_87_f,accuracy],feed_dict={x:batch_test_x,y:batch_test_y})
                print(result.shape)
                show_test_mask(result,batch_test_x)
                #print(resnext1.shape)
                show_img(resnext1[0,:,:,1])
                print('val acc',val_acc)
                    
                print('train ',ls,acc)
                saver.save(sess,adam_meta,global_step=i) 
                print('---------')
        coord.request_stop()
        coord.join(threads)
    