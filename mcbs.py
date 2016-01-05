import tensorflow as tf
import numpy as np
import time

mcPaths = 10000

spotDatas = [[3279.42]]
strikeDatas = [[3250]]
premiumCallDatas = [[245.8]]
premiumPutDatas = [[290.3]]

spot = tf.placeholder("float", [None,1], "spot")
strike = tf.placeholder("float", [None,1], "strike")
premiumCallMkt = tf.placeholder("float", [None,1], "premium_call_market")
premiumPutMkt = tf.placeholder("float", [None,1], "premium_put_market")
drift = tf.Variable(0.01*tf.random_normal([1,1]))
vol = tf.exp(tf.Variable(0.01*tf.random_normal([1,1])))
#gaussian = tf.random_normal([1,mcPaths])
gaussian = tf.constant(np.random.normal(size=(mcPaths)),dtype="float",shape=[1,mcPaths])
t = 1.0

driftPart = tf.mul(tf.sub(drift,tf.div(tf.mul(vol,vol),2.0)), t)
volPart = tf.mul(tf.mul(vol,gaussian), tf.sqrt(t))
spotMdl = tf.matmul(spot, tf.exp(tf.add(driftPart, volPart)))
forward = tf.mul(spot, tf.exp(driftPart))
premiumCallMdl = tf.maximum(tf.sub(spotMdl, strike),0.0)
premiumPutMdl = tf.maximum(tf.sub(strike, spotMdl),0.0)
premiumCallMc = tf.reduce_mean(premiumCallMdl,1)
premiumPutMc = tf.reduce_mean(premiumPutMdl,1)
error = tf.log(tf.reduce_sum(tf.pow(premiumCallMkt-premiumCallMc,2)+tf.pow(premiumPutMkt-premiumPutMc,2)))

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 200, 0.96, staircase=True)
optimiser = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(error, global_step=global_step)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
feed_dict = {spot: spotDatas, strike: strikeDatas, premiumCallMkt: premiumCallDatas, premiumPutMkt: premiumPutDatas}

print("drift: %.2f%%" % (sess.run(drift, feed_dict=feed_dict)[0][0]*100))
print("vol: %.2f%%" % (sess.run(vol, feed_dict=feed_dict)[0][0]*100))
print("premiumCallMc: %.2f" % sess.run(premiumCallMc, feed_dict=feed_dict)[0])
print("premiumPutMc: %.2f" % sess.run(premiumPutMc, feed_dict=feed_dict)[0])

clockCurrent = clockStart = time.clock()
numOld = 0
print "error, drift, vol, learning rate, mc/s (global), mc/s (current)"
for numCurrent in range(10000):
  res = sess.run([optimiser,error,drift,vol,learning_rate], feed_dict=feed_dict)
  if numCurrent and not (numCurrent%100):
    clockOld = clockCurrent
    clockCurrent = time.clock()
    print ','.join(map(str,(res[1],res[2][0][0],res[3][0][0],res[4],(clockCurrent-clockStart)/numCurrent,(clockCurrent-clockOld)/(numCurrent-numOld))))
    numOld = numCurrent
  
print("drift: %.2f%%" % (sess.run(drift, feed_dict=feed_dict)[0][0]*100))
print("vol: %.2f%%" % (sess.run(vol, feed_dict=feed_dict)[0][0]*100))
print("premiumCallMc: %.2f" % sess.run(premiumCallMc, feed_dict=feed_dict)[0])
print("premiumPutMc: %.2f" % sess.run(premiumPutMc, feed_dict=feed_dict)[0])