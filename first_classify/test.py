import tensorflow as tf
import os
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
test_size = 1


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
    
def show_test_mask(result,batch_x):
    result2 = np.reshape(result,(-1,64,64,2))
    mask = np.zeros(shape=[result2.shape[0],64,64],dtype=np.float32)
    for s in range(test_size):
        for i in range(64):
            for j in range(64):
                if(result2[s][i][j][1]>=0.7):
                    mask[s][i][j] = result2[s][i][j][1]
    image = batch_x[0]
    mask = mask[0]
    show_img(image)
    show_img(mask)

def get_test_data(path):
    file_path = list()
    for root,dirs,files in os.walk(path):
        for file in files:
            if(file.endswith('jpg')):
                fp = os.path.join(root,file)
                file_path.append(fp)
    return file_path

if(__name__=='__main__'):
    image_list = get_test_data('./test')
    
    y = tf.placeholder(tf.float32,[None,64,64,class_num])
    neg_weight = tf.placeholder(tf.float32,[None,64,64])
    #resnet32 = resnet_v2.ResNet(class_num)
    #resnet32 = resnet_v2.ResNet(class_num)
    #fpn_predict = resnet32.unconv5
    #resnet32 = dense_net.Dense_net(img_size,channel,class_num,24,0.5)
    #resnet32 = at_resnext.Resnext(img_size,channel,class_num,False)
    resnet32 = code87_net.Resnext(img_size,channel,class_num,False)
    x = resnet32.x
    fpn_predict = resnet32.code_87_pre
    
    with tf.Session() as sess:
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
        for i in range(100):
           img = cv2.imread(image_list[i])
           img = img[np.newaxis,:,:,:]
           result,resnext1 = sess.run([resnet32.code87_softmax,resnet32.code_87_f],feed_dict={x:img})
           show_test_mask(result,img)
           #print(resnext1.shape)
           show_img(resnext1[0,:,:,0])
            
        coord.request_stop()
        coord.join(threads)
    