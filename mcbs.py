import tensorflow as tf
import numpy as np

mcPaths = 10000

spotDatas = [[3279.42]]
strikeDatas = [[3250]]
premiumCallDatas = [[245.8]]
premiumPutDatas = [[290.3]]

spot = tf.placeholder("float", [None,1], "spot")
strike = tf.placeholder("float", [None,1], "strike")
premiumCallMkt = tf.placeholder("float", [None,1], "premium_call_market")
premiumPutMkt = tf.placeholder("float", [None,1], "premium_put_market")
drift = tf.Variable(0.0*tf.random_normal([1,1]))
vol = tf.exp(tf.Variable(0.01+0.0*tf.random_normal([1,1])))
gaussian = tf.random_normal([1,mcPaths])
t = 1.0

driftPart = tf.mul(tf.sub(drift,tf.div(tf.mul(vol,vol),2.0)), t)
volPart = tf.mul(tf.mul(vol,gaussian), tf.sqrt(t))
spotMdl = tf.matmul(spot, tf.exp(tf.add(driftPart, volPart)))
forward = tf.mul(spot, tf.exp(driftPart))
premiumCallMdl = tf.maximum(tf.sub(spotMdl, strike),0.0)
premiumPutMdl = tf.maximum(tf.sub(strike, spotMdl),0.0)
premiumCallMc = tf.reduce_mean(premiumCallMdl,1)
premiumPutMc = tf.reduce_mean(premiumPutMdl,1)

error = tf.reduce_sum(tf.sqrt(tf.pow(premiumCallMkt-premiumCallMc,2))+tf.sqrt(tf.pow(premiumPutMkt-premiumPutMc,2)))

optimiser = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(error)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
feed_dict = {spot: spotDatas, strike: strikeDatas, premiumCallMkt: premiumCallDatas, premiumPutMkt: premiumPutDatas}
print(sess.run(drift, feed_dict=feed_dict))
print(sess.run(vol, feed_dict=feed_dict))
print(sess.run(error, feed_dict=feed_dict))
print(sess.run(premiumCallMc, feed_dict=feed_dict))
print(sess.run(premiumPutMc, feed_dict=feed_dict))
for i in range(10000):
  print(sess.run([optimiser,error,drift,vol], feed_dict=feed_dict))
