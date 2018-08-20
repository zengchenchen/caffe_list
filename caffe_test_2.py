
# coding: utf-8

# In[1]:

from pylab import *
get_ipython().magic(u'matplotlib inline')


# In[2]:

caffe_root = '../' # 相对路径, python与caffe在同一路径下
import sys
sys.path.insert(0, caffe_root + 'python')


# In[3]:

import os
os.chdir(caffe_root) # 修改当前目录为 caffe_root

# 下载训练数据与验证数据
get_ipython().system(u'data/mnist/get_mnist.sh # \u7edd\u5bf9\u8def\u5f84\u662f\uff5e/ZZ/caffe/data/mnist/get_mnist.sh')
get_ipython().system(u'examples/mnist/create_mnist.sh # \u6d4b\u8bd5\u96c6')
os.chdir('examples') # 现在的相对路径在 ～/ZZ/caffe/examples下面


# In[4]:

# params里面有什么？
import caffe
from caffe import layers as L, params as P

#前一个n.**是后一个的输入，lenet函数是为了生成prototype
def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    
    # lmdb, Leveldb是键/值对（Key/Value Pair）嵌入式数据库管理系统编程库
    # transform_param数据预处理过程，scale:特征收缩系数 ntop:输出数量，本文是data和label，所以是2
    # n.label??
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2) 
    
    # weight_filler {type: "gaussian" std: 0.01}权重初始化;type='xavier',从[-scale, +scale]中进行均匀采样
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    
    # in-place计算能节约内存，在ReLU层，Dropout层，BatchNorm层，Scale层支持in-place计算
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    
    # n.score实际上还是全连接层，而且没有用到soft Max，看来不是之前理解的score
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    
    # n.loss到底是什么？？，n.label
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    
    return n.to_proto() # 返回net的prototype框架

# 函数lenet除了搭建一个net，还有有数据输入(source)，data_params自动生成
# 数据载入：net.blobs['data'].data[...] = transformed_image或者载入lmdb文件
lenet('mnist/mnist_train_lmdb', 64) 


# In[5]:

# 在～/ZZ/caffe/mnist生成了两个文件
# 为何生成两个prototxt呢？？
# 原因在于训练集与测试集的net框架都是一样的，唯一区别在于输入数据不一样
# 训练集mnist/mnist_train_lmdb，测试集mnist/mnist_test_lmdb
# 两个net的参数(权重，偏置，学习速率)等是共享的，因为solver = caffe.SGDSolver('mnist/lenet_auto_solver.prototxt')
with open('mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_train_lmdb', 64)))

with open('mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_test_lmdb', 100)))


# In[6]:

# learning parameters，如训练次数，各个参数的值(梯度下降设计的参数)
# 而layer对应的参数，在各自的layer里设置
get_ipython().system(u'cat mnist/lenet_auto_solver.prototxt')


# In[ ]:

# ## solver说明 ##

# # The train/test net protocol buffer definition
# train_net: "mnist/lenet_auto_train.prototxt"
# test_net: "mnist/lenet_auto_test.prototxt"
# # test_iter specifies how many forward passes the test should carry out.
# # In the case of MNIST, we have test batch size 100 and 100 test iterations,
# # covering the full 10,000 testing images.
# test_iter: 100 # mnist10000张图，batch_size=100,所以iteration=100次
# # Carry out testing every 500 training iterations.
# test_interval: 500 # 每训练500次测试一次
# # The base learning rate, momentum and the weight decay of the network.
# base_lr: 0.01 # base_lr用于设置基础学习率
# momentum: 0.9
# weight_decay: 0.0005 # 权重衰减项，防止过拟合的一个参数
# # The learning rate policy
# lr_policy: "inv" # lr调整策略
# gamma: 0.0001
# power: 0.75
# # Display every 100 iterations
# display: 100 # 每训练100次，在屏幕上显示一次。如果设置为0，则不显示
# # The maximum number of iterations
# max_iter: 10000 # 最大迭代次数
# # snapshot intermediate results
# snapshot: 5000 # 将训练出来的model和solver状态进行保存，snapshot用于设置训练多少次后进行保存，默认为0，不保存
# snapshot_prefix: "mnist/lenet" # snapshot_prefix设置保存路径。


# In[8]:

caffe.set_device(0) # 单显卡
caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver('mnist/lenet_auto_solver.prototxt') 


# In[9]:

# 说明建立layer‘data’时，通过source: "mnist/mnist_train_lmdb"把训练集的数据喂进去了！ 
solver.net.blobs['label'].data[:8]


# In[10]:

# each output is (batch size, feature dim, spatial dim)
[(k, v.data.shape) for k, v in solver.net.blobs.items()]


# In[11]:

# just print the weight sizes (we'll omit the biases)
[(k, v[0].data.shape) for k, v in solver.net.params.items()]


# In[12]:

# 训练集前向传播
# end='score' 现实score层的结果
solver.net.forward()  # train net


# In[13]:

# solver.net 与 solver.test_nets 两个！
# 测试集前向传播，0表示第一个net 
solver.test_nets[0].forward()  # test net (there can be more than one)


# In[14]:

# we use a little trick to tile the first eight images
# solver.net.blobs['data'].shape=(64, 1, 28, 28)
# solver.net.blobs['data'].data[:8, 0]=(8, 28, 28)
# 训练集data前八个视觉
imshow(solver.net.blobs['data'].data[:8, 0].transpose(1,0,2).reshape(28, 8*28), cmap='gray'); axis('off')

# 训练集label标签前八个，并非模型输出结果
print 'train labels:', solver.net.blobs['label'].data[:8]


# In[15]:

imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off')
print 'test labels:', solver.test_nets[0].blobs['label'].data[:8]


# In[16]:

solver.step(100) # 三步骤(前向传播，反向传播，更新权重)，进行了100遍
solver.test_nets[0].forward() # 果然，测试集的损失函数比之前小很多很多


# In[17]:

# solver.net.params['conv1'][0]=(20, 1, 5, 5)，20个filters 
# 每个filter由5*5个权重参数组成，将权重可视化
# diff为梯度信息？？？
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
       .transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray'); axis('off')


# In[18]:

solver.net.params['conv1'][0].data.shape


# In[22]:

get_ipython().run_cell_magic(u'time', u'', u"niter = 200 # \u4e00\u5171\u8bad\u7ec3\u4e86200\u6b21\uff0c\u8bad\u7ec3batch_size=64\uff0c\u6d4b\u8bd5batch_size=100\ntest_interval = 25 # \u6bcf\u8bad\u7ec325\u6b21\u5c31\u8fdb\u884c\u4e00\u6b21\u6d4b\u8bd5\n# losses will also be stored in the log\ntrain_loss = zeros(niter) # train_loss.shape=(200,)\ntest_acc = zeros(int(np.ceil(niter / test_interval))) # np.ceil(0.1)=1, test_acc.shape=(8,)\noutput = zeros((niter, 8, 10))\n\n# the main solver loop\nfor it in range(niter):\n    solver.step(1)  # SGD by Caffe\n    \n    # store the train loss\n    train_loss[it] = solver.net.blobs['loss'].data\n    \n    # store the output on the first test batch\n    # (start the forward pass at conv1 to avoid loading new data)\n    solver.test_nets[0].forward(start='conv1')\n    \n    # \u5148forward\u624d\u6709score\n    # solver.test_nets[0].blobs['score'].data.shape=(100, 10)\n    output[it] = solver.test_nets[0].blobs['score'].data[:8] \n    \n    # run a full test every so often\n    # (Caffe can also do this for us and write to a log, but we show here\n    #  how to do it directly in Python, where more complicated things are easier.)\n    if it % test_interval == 0:\n        print 'Iteration', it, 'testing...'\n        correct = 0\n        for test_it in range(100):\n            solver.test_nets[0].forward()\n            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)\n                           == solver.test_nets[0].blobs['label'].data)\n        test_acc[it // test_interval] = correct / 1e4")


# In[23]:

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))


# In[21]:

for i in range(8):
    figure(figsize=(2, 2))
    imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    figure(figsize=(10, 2))
    imshow(output[:50, i].T, interpolation='nearest', cmap='gray')
    xlabel('iteration')
    ylabel('label')


# In[25]:

train_net_path = 'mnist/custom_auto_train.prototxt'
test_net_path = 'mnist/custom_auto_test.prototxt'
solver_config_path = 'mnist/custom_auto_solver.prototxt'

### define net
def custom_net(lmdb, batch_size):
    # define your own net!
    n = caffe.NetSpec()
    
    # keep this data layer for all networks
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    
    # EDIT HERE to try different networks
    # this single layer defines a simple linear classifier
    # (in particular this defines a multiway logistic regression)
    n.score =   L.InnerProduct(n.data, num_output=10, weight_filler=dict(type='xavier'))
    
    # EDIT HERE this is the LeNet variant we have already tried
    # n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    # n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    # n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    # EDIT HERE consider L.ELU or L.Sigmoid for the nonlinearity
    # n.relu1 = L.ReLU(n.fc1, in_place=True)
    # n.score =   L.InnerProduct(n.fc1, num_output=10, weight_filler=dict(type='xavier'))
    
    # keep this loss layer for all networks
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    
    return n.to_proto()
with open(train_net_path, 'w') as f:
    f.write(str(custom_net('mnist/mnist_train_lmdb', 64)))    
with open(test_net_path, 'w') as f:
    f.write(str(custom_net('mnist/mnist_test_lmdb', 100)))

### define solver
from caffe.proto import caffe_pb2
s = caffe_pb2.SolverParameter()

# Set a seed for reproducible experiments:
# this controls for randomization in training.
s.random_seed = 0xCAFFE

# Specify locations of the train and (maybe) test networks.
s.train_net = train_net_path
s.test_net.append(test_net_path)
s.test_interval = 500  # Test after every 500 training iterations.
s.test_iter.append(100) # Test on 100 batches each time we test.

s.max_iter = 10000     # no. of times to update the net (training iterations)
 
# EDIT HERE to try different solvers
# solver types include "SGD", "Adam", and "Nesterov" among others.
s.type = "SGD"

# Set the initial learning rate for SGD.
s.base_lr = 0.01  # EDIT HERE to try different learning rates
# Set momentum to accelerate learning by
# taking weighted average of current and previous updates.
s.momentum = 0.9
# Set weight decay to regularize and prevent overfitting
s.weight_decay = 5e-4

# Set `lr_policy` to define how the learning rate changes during training.
# This is the same policy as our default LeNet.
s.lr_policy = 'inv'
s.gamma = 0.0001
s.power = 0.75
# EDIT HERE to try the fixed rate (and compare with adaptive solvers)
# `fixed` is the simplest policy that keeps the learning rate constant.
# s.lr_policy = 'fixed'

# Display the current training loss and accuracy every 1000 iterations.
s.display = 1000

# Snapshots are files used to store networks we've trained.
# We'll snapshot every 5K iterations -- twice during training.
s.snapshot = 50
s.snapshot_prefix = 'mnist/custom_net11'

# Train on the GPU
s.solver_mode = caffe_pb2.SolverParameter.GPU

# Write the solver to a temporary file and return its filename.
with open(solver_config_path, 'w') as f:
    f.write(str(s))

### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.get_solver(solver_config_path)

### solve
niter = 250  # EDIT HERE increase to train for longer
test_interval = niter / 10
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))


# In[ ]:



