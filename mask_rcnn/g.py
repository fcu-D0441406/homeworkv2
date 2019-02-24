import tensorflow as tf
import numpy as np

x = np.array([[[[0,0,0,0],[6,1,3,1],[3,4,5,6]]
            ,[[0,0,0,0],[0,0,0,0],[3,1,8,7]]]])

x2 = np.array([[[[1,1,2,1],[5,1,3,1],[1,2,3,4]]
            ,[[1,4,1,5],[1,6,1,2],[2,3,4,5]]]])

j = tf.placeholder(tf.float32,[None,1,2,3,4])
j2 = tf.placeholder(tf.float32,[None,1,2,3,4])

loss = 0
j3 = tf.add(j,-1*j2)
j4 = tf.cast(tf.equal(j,0),tf.float32)
j5 = (1-j4)*j3
j6 = tf.reduce_sum(j5,axis=4)
j7 = tf.cast(tf.less(j6,1.0),tf.float32)
j8 = (j7)*(0.5*j6*j6)+(1.0-j7)*(tf.abs(j6)-0.5)
j9 = tf.reduce_sum(j8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x = x[np.newaxis,:,:,:,:]
    x2 = x2[np.newaxis,:,:,:,:]
    xx0,xx,xx2,xx3,xx4,xx5 = sess.run([j3,j5,j6,j7,j8,j9],feed_dict={j:x,j2:x2})
    print(xx0)
    print('-----')
    print(xx)
    print('-----')
    print(xx2.shape)
    print('-----')
    print(xx3)
    print('-----')
    print(xx4)
    print('-----')
    print(xx5)


#a = tf.placeholder(tf.float32)
#print(a)

'''
a = [1,2,3]
b,c,d = a[:]
print(b,c,d)
'''
