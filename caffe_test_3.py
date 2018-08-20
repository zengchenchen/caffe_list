
# coding: utf-8

# In[41]:

##直接拿网上已经训练好的模型来用，在它的基础上微调参数##
# 微调注意事项：如果自己数据的种类与原始种类不一样，那么就要重新训练最后分类全连接层，也就是重新再来！其他层的尺寸等等可以不变！ #



caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

import numpy as np
from pylab import *
get_ipython().magic(u'matplotlib inline')
import tempfile # 生成零时文件，关闭即删除

# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image


# In[42]:

# Download just a small subset of the data for this exercise.
# (2000 of 80K images, 5 of 20 labels.)
# To download the entire dataset, set `full_dataset = True`.
full_dataset = False
if full_dataset:
    NUM_STYLE_IMAGES = NUM_STYLE_LABELS = -1
else:
    NUM_STYLE_IMAGES = 2000
    NUM_STYLE_LABELS = 5

# This downloads the ilsvrc auxiliary data (mean file, etc),
# and a subset of 2000 images for the style recognition task.
# 把图片和标签保存在哪里了，是分开保存吗？--只下载1000张图
import os
os.chdir(caffe_root)# run scripts from caffe root
get_ipython().system(u'data/ilsvrc12/get_ilsvrc_aux.sh')
get_ipython().system(u'scripts/download_model_binary.py models/bvlc_reference_caffenet')
get_ipython().system(u'python examples/finetune_flickr_style/assemble_data.py     --workers=-1  --seed=1701     --images=$NUM_STYLE_IMAGES  --label=$NUM_STYLE_LABELS')
# back to examples
os.chdir('examples')


# In[43]:

import os
# os.path.join(a,b,a) = /a/b/c
# weight路径为caffemodel路径
weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
assert os.path.exists(weights)


# In[44]:

# Load ImageNet labels to imagenet_labels
# 下载label，synset_words.txt 如 n07739125 apple
imagenet_label_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))
assert len(imagenet_labels) == 1000
print 'Loaded ImageNet labels:\n', '\n'.join(imagenet_labels[:10] + ['...'])

# Load style labels to style_labels
# style_names.txt保存形容词，如阳光、忧郁、开心等等
style_label_file = caffe_root + 'examples/finetune_flickr_style/style_names.txt'
style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
if NUM_STYLE_LABELS > 0:
    style_labels = style_labels[:NUM_STYLE_LABELS]
print '\nLoaded style labels:\n', ', '.join(style_labels)


# In[47]:

from caffe import layers as L
from caffe import params as P

# lr_mult:学习率的系数， 偏置的lr_mult一般是权重的两倍
# decay_mult 衰退的系数
weight_param = dict(lr_mult=1, decay_mult=1) 
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param] # 要学习的参数是权重和偏置

## 问题 CNN中的frozen是什么？？ ##
frozen_param = [dict(lr_mult=0)] * 2

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    
    # L.Relu(conv, in_place=True) L.Relu()函数输入是conv
    return conv, L.ReLU(conv, in_place=True) 

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# 函数caffenet中的train指的是是否开始训练，如果False则状态一只预测不训练即不反向传播
# learning_all=false则状态二训练但只训练最后分类层；learning_all=true则状态三分类层和卷积层都训练
def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data = data
    
    # frozen ?? #
    # 解释：如果只训练最后一层即全连接层，那么其他层的参数都不变也就是frozen
    param = learned_param if learn_all else frozen_param
    
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    
    ##  这个if else 目的是什么？ ##
    # 如果只预测不训练，params取多少都无所谓，因为不用反向传播，如果需要训练，那么就得分情况啦
    # 如果训练，则还要增加一个层：L.Dropout
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
        
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    
    # 对于只预测不训练，当然只是输出每种类型对应的可能结果啦
    if not train:
        n.probs = L.Softmax(fc8)
        
    # 对于要训练的，那么就是得输出训练集的损失函数和正确率啦
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        print(n.to_proto())
        return f.name


# In[ ]:

# 这是conv层的参数，当caffenet函数中train=True,learn_all=False时，即只训练分类层
# 则只有fc8全连接层的parame=learned_param其余的param=0
# 那么反向传播时，就可以做到其余层的权重偏置不变，只有全连接层的参数改变了

# layer {
#     name: "conv1"
#     type: "Convolution"
#     bottom: "data"
#     top: "conv1"
#     # learning rate and decay multipliers for the filters
#     param { lr_mult: 1 decay_mult: 1 }   ***fc8全连接层的parame
#     # learning rate and decay multipliers for the biases
#     param { lr_mult: 2 decay_mult: 0 }   ***fc8全连接层的parame
#     convolution_param {
#       num_output: 96     # learn 96 filters
#       kernel_size: 11    # each filter is 11x11
#       stride: 4          # step 4 pixels between each filter application
#       weight_filler {
#         type: "gaussian" # initialize the filters from a Gaussian
#         std: 0.01        # distribution with stdev 0.01 (default mean: 0)
#       }
#       bias_filler {
#         type: "constant" # initialize the biases to zero (0)
#         value: 0
#       }
#     }
#   }


# In[48]:

# DummyData:可以用这一层模拟预测过程，只预测不训练
# 想一想为什么用L.DummyData来代替只预测不训练的net里的data
dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
imagenet_net_filename = caffenet(data=dummy_data, train=False)

# n_proto()可以观察net的结构，imagenet_net可以认为是一个完整的预测模型了(caffe.TEST),且没有反向传播过程
# 注意到imagenet_net没有source即数据来源，因为它是预测模型，只需要输入待预测的数据即可，disp_imagenet_preds(imagenet_net, image)
imagenet_net = caffe.Net(imagenet_net_filename, weights, caffe.TEST)


# In[49]:

# style_net函数对应的是只训练最后分类层参数
# 作用跟caffeine一样，return的也是caffenet
# 区别在于data=L.ImageData(),style_net有数据输入(data与label)以及在Data层增加了数据处理部分transform_param
# subset的作用就是决定source是train还是test
def style_net(train=True, learn_all=False, subset=None):
    if subset is None:
        subset = 'train' if train else 'test'
        
    # 'data/flickr_style/train.txt'存储内容：/home/momo/ZZ/caffe/data/flickr_style/images/10344996196_1117743cfe.jpg 0
    # /home/momo/ZZ/caffe/data/flickr_style/images/10344996196_1117743cfe.jpg是style_data, 0是style_label
    # L.ImagData()中的source直接输入图，不是lmdb文件
    source = caffe_root + 'data/flickr_style/%s.txt' % subset
    
    # mirror:是否对输入数据采取随机水平镜像，增加训练数据
    # crop_size:尺寸超过277就会进行裁剪
    # mean_file: 数据均值文件存放
    transform_param = dict(mirror=train, crop_size=227,
        mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    
    style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=50, new_height=256, new_width=256, ntop=2)
    return caffenet(data=style_data, label=style_label, train=train,
                    num_classes=NUM_STYLE_LABELS,
                    classifier_name='fc8_flickr',
                    learn_all=learn_all)

print(style_net(train=False, subset='train'))


# In[50]:

untrained_style_net = caffe.Net(style_net(train=False, subset='train'),
                                weights, caffe.TEST)
untrained_style_net.forward()
style_data_batch = untrained_style_net.blobs['data'].data.copy()
style_label_batch = np.array(untrained_style_net.blobs['label'].data, dtype=np.int32)
style_data_batch


# In[9]:

# 预测函数
def disp_preds(net, image, labels, k=5, name='ImageNet'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image # 这一步是喂数据的步骤！
    probs = net.forward(start='conv1')['probs'][0] 
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted %s labels =' % (k, name)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))

# 只预测不训练
def disp_imagenet_preds(net, image):
    disp_preds(net, image, imagenet_labels, name='ImageNet')

# 只对最后分类层的参数进行训练
def disp_style_preds(net, image):
    disp_preds(net, image, style_labels, name='style')


# In[10]:

batch_index = 8
image = style_data_batch[batch_index]
plt.imshow(deprocess_net_image(image))
print 'actual label =', style_labels[style_label_batch[batch_index]]


# In[53]:

image.shape


# In[11]:

disp_imagenet_preds(imagenet_net, image)


# In[12]:

disp_style_preds(untrained_style_net, image)


# In[13]:

diff = untrained_style_net.blobs['fc7'].data[0] - imagenet_net.blobs['fc7'].data[0]
error = (diff ** 2).sum()
assert error < 1e-8


# In[14]:

del untrained_style_net


# In[15]:

from caffe.proto import caffe_pb2

def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    s.snapshot_prefix = caffe_root + 'models/finetune_flickr_style/finetune_flickr_style'
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name


# In[16]:

def run_solvers(niter, solvers, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights


# In[17]:

niter = 200  # number of iterations to train

# Reset style_solver as before.
style_solver_filename = solver(style_net(train=True))
style_solver = caffe.get_solver(style_solver_filename)
style_solver.net.copy_from(weights)

# For reference, we also create a solver that isn't initialized from
# the pretrained ImageNet weights.
scratch_style_solver_filename = solver(style_net(train=True))
scratch_style_solver = caffe.get_solver(scratch_style_solver_filename)

print 'Running solvers for %d iterations...' % niter
solvers = [('pretrained', style_solver),
           ('scratch', scratch_style_solver)]
loss, acc, weights = run_solvers(niter, solvers)
print 'Done.'

train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
style_weights, scratch_style_weights = weights['pretrained'], weights['scratch']

# Delete solvers to save memory.
del style_solver, scratch_style_solver, solvers


# In[18]:

plot(np.vstack([train_loss, scratch_train_loss]).T)
xlabel('Iteration #')
ylabel('Loss')


# In[19]:

plot(np.vstack([train_acc, scratch_train_acc]).T)
xlabel('Iteration #')
ylabel('Accuracy')


# In[20]:

def eval_style_net(weights, test_iters=10):
    test_net = caffe.Net(style_net(train=False), weights, caffe.TEST)
    accuracy = 0
    for it in xrange(test_iters):
        accuracy += test_net.forward()['acc']
    accuracy /= test_iters
    return test_net, accuracy


# In[21]:

test_net, accuracy = eval_style_net(style_weights)
print 'Accuracy, trained from ImageNet initialization: %3.1f%%' % (100*accuracy, )
scratch_test_net, scratch_accuracy = eval_style_net(scratch_style_weights)
print 'Accuracy, trained from   random initialization: %3.1f%%' % (100*scratch_accuracy, )


# In[22]:

end_to_end_net = style_net(train=True, learn_all=True)

# Set base_lr to 1e-3, the same as last time when learning only the classifier.
# You may want to play around with different values of this or other
# optimization parameters when fine-tuning.  For example, if learning diverges
# (e.g., the loss gets very large or goes to infinity/NaN), you should try
# decreasing base_lr (e.g., to 1e-4, then 1e-5, etc., until you find a value
# for which learning does not diverge).
base_lr = 0.001

style_solver_filename = solver(end_to_end_net, base_lr=base_lr)
style_solver = caffe.get_solver(style_solver_filename)
style_solver.net.copy_from(style_weights)

scratch_style_solver_filename = solver(end_to_end_net, base_lr=base_lr)
scratch_style_solver = caffe.get_solver(scratch_style_solver_filename)
scratch_style_solver.net.copy_from(scratch_style_weights)

print 'Running solvers for %d iterations...' % niter
solvers = [('pretrained, end-to-end', style_solver),
           ('scratch, end-to-end', scratch_style_solver)]
_, _, finetuned_weights = run_solvers(niter, solvers)
print 'Done.'

style_weights_ft = finetuned_weights['pretrained, end-to-end']
scratch_style_weights_ft = finetuned_weights['scratch, end-to-end']

# Delete solvers to save memory.
del style_solver, scratch_style_solver, solvers


# In[23]:

test_net, accuracy = eval_style_net(style_weights_ft)
print 'Accuracy, finetuned from ImageNet initialization: %3.1f%%' % (100*accuracy, )
scratch_test_net, scratch_accuracy = eval_style_net(scratch_style_weights_ft)
print 'Accuracy, finetuned from   random initialization: %3.1f%%' % (100*scratch_accuracy, )


# In[24]:

plt.imshow(deprocess_net_image(image))
disp_style_preds(test_net, image)


# In[25]:

batch_index = 1
image = test_net.blobs['data'].data[batch_index]
plt.imshow(deprocess_net_image(image))
print 'actual label =', style_labels[int(test_net.blobs['label'].data[batch_index])]


# In[26]:

disp_style_preds(test_net, image)


# In[27]:

disp_style_preds(scratch_test_net, image)


# In[28]:

disp_imagenet_preds(imagenet_net, image)

