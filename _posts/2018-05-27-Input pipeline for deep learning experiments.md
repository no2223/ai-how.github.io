---
layout: post
title: Input pipeline for deep learning experiments: Keras, Tensorflow and Pytorch 
published: false
comments: true
---

Increasing list of algorithms and techniques aiming to improve the performance of deep learning models often instills a curiosity to benchmark how well these models perform. Benchmarking these techniques (on a dataset specific to business) often require writing your own pipeline which could quickly fetch mini-batches and run multiple iterations to search for optimal hyper parameters.

A quick and dirty practice is to load your training data into RAM using numpy and pandas functionalities (np.laod or pd.read_csv). This works well only if the dataset is small enough to fit in the memory. From a personal experience this slows down the entire training process resulting in longer model development and evaluation cycle. This blog post describes how you can quickly write input pipeline on a platform of your own choice:

# Keras
keras has been the standard choice to start with deep learning experiments as it avoid understanding all the nitty gritty details and provides a high level API to build model. For any deep learning experiment the training involves updating the model weights by estimating the gradients of loss w.r.t model hyperparameters. To quickly iterate through the data requires how fast the mini-batches can be fetched and run on GPU for loss and gradient computation. Numpy offers the functionality to read the data from disk without loading into RAM (a great relief !! as it frees the space for other processing). Code snippet below details how mini-batches are fetched:

```python
##=============================================##
##  data iterator reads data from disk ========##
## and yields mini-batches for weight updation=##
##=============================================##

batch_size =    # defines the mini-batch size

def train_data_gen():
    while 1:
        feat_path = "/path/to/feature/file/*.npy"
        label_path = "/path/to/label/file/*.npy"
        x = np.load(feat_path, mmap_mode='r')     # points to data location in memory mapped mode
        y = np.load(label_path, mmap_mode='r')    # reduced space requirement since data is not loaded in RAM
        lst = range(x.shape[0])
        ##=====================================##
        ## index shuffling after each epoch====##
        ##=====================================##
        shuffle(lst)
        iters = len(lst)/batch_size
        print (iters)
        for i in range(iters):
            # create numpy arrays of input data
            # and labels, from each line in the file
            #print (len(lst[(i*batch_size):((i+1)*batch_size)]))
            yield (x[lst[(i*batch_size):((i+1)*batch_size)]], y[lst[(i*batch_size):((i+1)*batch_size)]])
```

Note: Avoid loading the data into RAM and read data from disk on an iteration basis. However when using this iterator with Keras .fit_generator queues the mini-batches for seamless training without requiring an optimizer to wait for next batch

# Tensorflow

Tensorflow offers queuing mechanisms which are thread safe helps implement queues improving the time required for fetching mini-batch. Below GIF illustrates the queuing.

<p align="center"> <img src="https://www.tensorflow.org/images/IncremeterFifoQueue.gif" width="450" height="300" /> </p>

```python

##=======================================================##
##  data loader reads data from disk ====================##
## fills the queue and yields mini-batch of desried size=##
##=======================================================##

class ThreadRunner(object):
    """
    This class manages the queuing and dequeuing.
    """
    def __init__(self):
        self.feat = tf.placeholder(dtype=tf.float32, shape=[None,25,25])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None,])
        self.name = tf.placeholder(dtype=tf.string, shape=[None,])
        # queue defined using tensorflow holds the fetaures and labels equal to capacity defined
        self.queue = tf.RandomShuffleQueue(shapes=[[25,25],[25,25],[]],
                                           dtypes=[tf.float32,tf.int32,tf.string],
                                           capacity=1384,
                                           min_after_dequeue=1000)
        # filling queue using enqueue_many operations
        self.enqueue_op = self.queue.enqueue_many([self.feat,self.labels,self.name])
    def get_inputs(self):
        """
        Return's tensors containing a batch of feature sets and labels of size 32
        """
        features, labels, name = self.queue.dequeue_many(32)
        return features, labels, name
    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for features, labels, name in data_iterator():
            sess.run(self.enqueue_op, feed_dict={self.feat:features,self.labels:labels, self.name:name})
    def start_threads(self, sess, n_threads=2):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            #print "thread started :", n
            sys.stdout.flush()
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
        
##=======================================================##
## restricts the queuing process to cpu==================##
##=======================================================##

with tf.device("/cpu:0"):
    run_thread = ThreadRunner()
    features, labels, name = run_thread.get_inputs()

##=======================================================##
## initiating the session and starting threads===========##
##=======================================================##
sess = tf.Session()
tf.train.start_queue_runners(sess=sess)
custom_runner.start_threads(sess)
print "thread started :"

##=======================================================##
## to load the data simply run sess as below=============##
##=======================================================##

feat_X, Y = sess.run([features, labels, name])

##=======================================================##
## call optimizer to run on mini-batch===================##
##=======================================================##

sess.run([train_opt], feed_dict = {X: feat_X, Y: Y})
```

# Pytorch
