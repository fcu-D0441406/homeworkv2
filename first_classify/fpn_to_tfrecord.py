import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.cross_validation import train_test_split
import random
import matplotlib.pyplot as plt

tfrecord_train = './fpn_train.tfrecords'
tfrecord_test = './fpn_test.tfrecords'
original_img = 'img.png'
mask_img = 'label.png'
channel = 3
img_size = 256
class_num = 2

def get_data(file_path):
    image_list = []
    label_list = []
    for root,dirs,files in os.walk(file_path):
        d = [ os.path.join(root,di) for di in dirs]
        for dx in d:
            image_list.append(os.path.join(dx,original_img))
            label_list.append(os.path.join(dx,mask_img))
    return image_list,label_list

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(images_list,labels_list,save_dir,name):
    tfrecord_filename = os.path.join(save_dir,name+'.tfrecords')
    n_samples = len(labels_list)
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    print('\nTransform start')
    for i in np.arange(0,n_samples):
        if(channel==1):
            image = cv2.imread(images_list[i],0)
        elif(channel==3):
            image = cv2.imread(images_list[i])
            '''
            cv2.imshow('img',image)
            cv2.waitKey()
            cv2.destroyAllWindows()
            '''
        try:
            #print(image.shape,image_list[i])
            image = np.reshape(image,(img_size,img_size,channel))
            image_raw = image.tostring()
            #print(len(image_raw))
            #label = int(labels_list[i])
            label = convert_label(label_list[i])
            label = label.astype(np.uint8)
            #print(label.shape)
            label_raw = label.tostring()
            #print(len(label_raw))
            example = tf.train.Example(features=tf.train.Features(feature={'label_raw':bytes_feature(label_raw),
                                                                           'image_raw':bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except:
            print('no file')
    writer.close()
    print('transform susccessful')

def read_and_decode(tfrecord_file_path,batch_size):
    tfrecord_file = tf.train.string_input_producer([tfrecord_file_path])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(tfrecord_file)
    img_features = tf.parse_single_example(serialized_example,features={
                                        'label_raw':tf.FixedLenFeature([],tf.string),
                                        'image_raw':tf.FixedLenFeature([],tf.string),
                                        })
    image = tf.decode_raw(img_features['image_raw'],tf.uint8)
    image = tf.reshape(image,[img_size,img_size,3])
    label = tf.decode_raw(img_features['label_raw'],tf.uint8)

    label = tf.reshape(label,[64,64,2])
    image_batch,label_batch = tf.train.shuffle_batch([image,label],
                                                     batch_size=batch_size,
                                                     min_after_dequeue=1,
                                                     num_threads=64,
                                                     capacity=2)
    return image_batch,label_batch

def convert_label(label):
    label = cv2.imread(label)
    
    result = np.zeros([img_size,img_size,class_num])
    for i in range(0,img_size):
        for j in range(0,img_size):
            if(label[i][j][2]!=0):
                result[i][j][1] = 1
            else:
                result[i][j][0] = 1
    result = cv2.resize(result,(64,64))
    #print(result.shape)
    return result

def show_img(img):
    if(img.shape[-1]==1):
        plt.imshow(img,cmap ='gray')
        plt.show()
    else:
        plt.imshow(img)
        plt.show()

def debug(image_list,label_list):
    img = cv2.imread(label_list[0])
    for i in range(256):
        for j in range(256):
            if(img[i][j][2]!=0):
                print(img[i][j][2])
    print(img.shape)
    show_img(img)
    
    

if(__name__=='__main__'):
    print('start')
    image_list,label_list = get_data('./fpn_img')
    #debug(image_list,label_list)
    
    img_train,img_test,label_train,label_test = train_test_split(image_list,label_list,test_size = 0.1
                                                                 , random_state=random.randint(0,100))
    convert_to_tfrecord(img_train,label_train,'./','fpn_train')
    convert_to_tfrecord(img_test,label_test,'./','fpn_test')
    
    ig,lb = read_and_decode(tfrecord_train,1)
    print(ig.shape,lb.shape)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        batch_x = sess.run(ig)
        batch_y = sess.run(lb)
        batch_x = np.reshape(batch_x,(256,256,3))
        show_img(batch_x)
        coord.request_stop()
        coord.join(threads)