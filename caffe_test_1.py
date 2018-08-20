
# coding: utf-8

# In[ ]:


import numpy as np # 加载numpy
import matplotlib.pyplot as plt # 加载matplotlib
#get_ipython().magic(u'matplotlib inline # \u6b64\u5904\u662f\u4e3a\u4e86\u80fd\u5728notebook\u4e2d\u76f4\u63a5\u663e\u793a\u56fe\u50cf')

# rcParams是一个包含各种参数的字典结构，含有多个key-value，可修改其中部分值
plt.rcParams['figure.figsize'] = (10, 10) # 图像显示大小，单位是英寸 
plt.rcParams['image.interpolation'] = 'nearest'  # 最近邻差值,像素为正方形
plt.rcParams['image.cmap'] = 'gray'  # 使用灰度输出而不是彩色输出


# In[ ]:


import sys
caffe_root = '../'  # caffe根目录，此处为相对路径，如果失灵，可换成绝对路径

# sys.path是一个列表，insert()函数插入一行，也可以使用sys.path.append('模块地址')
sys.path.insert(0, caffe_root + 'python') # 加载caffe的python模块 ??? 
import caffe # 加载caffe
import os

# 如果该路径下存在caffemodel文件，则打印信息，否则从官网下载, cafemodel是CNN模型
if os.path.isfile('~/ZZ/caffe/zengchenchen/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    get_ipython().system(u'../scripts/download_model_binary.py ../models/bvlc_reference_caffenet')


# In[ ]:


caffe.set_mode_cpu() # 设置caffe为cpu模式，也可设成gpu模式(caffe.det_mode_gpu)
model_def = '~/ZZ/caffe/zengchenchen/deploy.prototxt' # 结构部署文件，存储CNNlayer信息, 每个layer类型都过一遍
model_weights = '~/ZZ/caffe/zengchenchen/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # 定义模型结构 
                model_weights,  # 包含模型训练权重，caffemodel保存参数，应该打开这个文件看一看
                caffe.TEST)     # 使用测试模式(训练中不能执行dropout) ,使用caffemodel的权重直接进行测试，都不用训练的，caffe.TEST


# In[ ]:


# 加载ImageNet训练集的图像均值，预处理需要减去均值,验证集也需要进行数据预处理步骤
# ilsvrc_2012_mean.npy文件是numpy格式，其数据维度是(3L, 256L, 256L)
mu = np.load('~/ZZ/caffe/zengchenchen/ilsvrc_2012_mean.npy') # 加载均值文件
mu = mu.mean(1).mean(1)  # 对所有像素值取平均以此获取BGR的均值像素值
print 'mean-subtracted values:', zip('BGR', mu) 
# 取平均后得到BGR均值分别是[104.00698793,116.66876762,122.67891434]

# transformer目的是对输入数据进行变换，所以用'data'，
# caffe.io.transformer是一个类，实体化的时候构造函数__init__(self, inputs)给一个初值
# 其中net.blobs本身是一个字典，每一个key对应每一层的名字，#net.blobs['data'].data.shape计算结果为(10, 3, 227, 227)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})



## 图片预处理过程：先定义好transformer，然后reshape‘data’，再load_image图片，最后transformer.preprocess图片##

# 以下都是caffe.io.transformer类的函数方法
#caffe.io.transformer的类定义放在io.py文件中，也可用help函数查看说明
transformer.set_transpose('data', (2,0,1))    # 将图像通道数设置为outermost的维数
transformer.set_mean('data', mu)              # 每个通道减去均值
transformer.set_raw_scale('data', 255)        # 像素值从[0,1]变换为[0,255]
transformer.set_channel_swap('data', (2,1,0)) # 交换通道，RGB->BGR


# In[ ]:


#设置输入图像大小
net.blobs['data'].reshape(50,       # 尽管只检测一张图片，batch size仍为50，之前默认的net.blobs['data'].data.shape=(10,3,227,227)
                          3,        # 3通道
                          227, 227) # 图片尺寸227x227


# In[ ]:


# 加载图片，函数声明为load_image(filename, color=True)
image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg') 
# 按照之前设置进行预处理
transformed_image = transformer.preprocess('data', image) 
plt.imshow(image) #显示图片


# In[ ]:


# 图片处理完了就是加载到net里啦～  将图像数据拷贝到为net分配的内存中
net.blobs['data'].data[...] = transformed_image 

# output={'prob':array([[]])}
# 前向传播，跑一遍网络，默认结果为最后一层的blob（也可以指定某一中间层），赋给output
output = net.forward() 

# output['prob']矩阵的维度是(50, 1000)
output_prob = output['prob'][0]  # 取batch中第一张图像的概率值
# 打印概率最大的类别代号，argmax()函数是求取矩阵中最大元素的索引
print 'predicted class is:', output_prob.argmax() 


# In[ ]:


# 加载ImageNet标签，如果不存在，则会自动下载
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    get_ipython().system(u'../data/ilsvrc12/get_ilsvrc_aux.sh')

# 读取纯文本数据，三个参数分别是文件地址、数据类型和数据分隔符，保存为字典格式    
labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()] 


# In[ ]:


# 从softmax output可查看置信度最高的五个结果
top_inds = output_prob.argsort()[::-1][:5]  # 逆序排列，取前五个最大值

print 'probabilities and labels:'
print(zip(output_prob[top_inds], labels[top_inds])) 


# In[ ]:


# 查看CPU的分类时间，然后再与GPU进行比较
print(%timeit net.forward()) #计时


for layer_name, blob in net.blobs.items(): # net.blob以字典形式记录layer_name与对应的数据(top数据，计算结果，可以认为是每个像素点的值)
    print(layer, blob.data)                # 例如最后一行输出(layer, blob.data.shape) = ('prob', [50, 1000])
                                           # 50-batch_size, 1000每个类型对应的概率
                                           # net.params['fc8'].data.shape


##问题，net.params的数据是怎么来的,训练好的模型保存在caffemodel文件里，里面就有权重偏置等等##
for layer_name, params in net.params.item():      # net.params={'fc8':object[[......],[...]],'conv1':[[],[]]}
    print(layer, params[0].data, params[1].data)  # net.params['fc8'][0].data.shape,有些layer没有params，例如'prob'                                




# 在prototype.txt文件中，反向传播涉及到的参数是哪些，还有自己调参的是哪些参数。这个文件为啥没有weight_filler {#type: "gaussian"，std: 0.01}
# 这个例子直接用测试集，所以框架里没有loss function
# 这个框架定义了layer以及其中的维度、尺寸或者卷积核个数等等，但是没有涉及到一些参数例如卷积层的权重和偏置的初始值

name: "CaffeNet"  
layer {
  name: "data"
  type: "Input"   # layer'data'没有bottom，数据是如何输入的：首先定义net = caffe.Net(model_def, model_weights, caffe.TEST)；然后net.blobs['data'].data[...] = transformed_image      
  top: "data"     # 其中'data'指的就是layer'data'
  input_param { shape: { dim: 10 dim: 3 dim: 227 dim: 227 } } # 输入没有参数，只有输入数据，输入维度[batch_size, channl, height, weight]
}
# layer {
#   name: "data"
#   type: "Data"
#   top: "data"
#   top: "label"
#   transform_param {
#     scale: 0.00392156862745
#   }
#   data_param {
#     source: "mnist/mnist_train_lmdb"  # 这也是一种加载数据如训练集的方式
#     batch_size: 64
#     backend: LMDB
#   }
# }
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param { # 卷积层涉及参数，没有填充数量p
    num_output: 96 # 卷积核个数
    kernel_size: 11 # 卷积核 11*11 
    stride: 4 # 步长
  }
}
layer {            # 激励函数没有超参数，且bottom与top都是conv1
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX # 最大池化
    kernel_size: 3 
    stride: 2
  }
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "pool1"
  top: "norm1"
  lrn_param {
    local_size: 5 # 第i个图像的第j个通道像素的第k个像素点，以它为中心，选择附近5个通道像素点
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "norm1"
  top: "conv2"
  convolution_param {
    num_output: 256
    pad: 2 # 填充2步，这里倒是设置零填充了
    kernel_size: 5
    group: 2  # 极端情况下，输入输出通道数相同，比如为24，group大小也为24，那么每个输出卷积核，只与输入的对应的通道进行卷积 
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm2"
  type: "LRN"
  bottom: "pool2"
  top: "norm2"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "norm2"
  top: "conv3"
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "conv4"
  type: "Convolution"
  bottom: "conv3"
  top: "conv4"
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 2
  }
}
layer {
  name: "relu4"
  type: "ReLU"
  bottom: "conv4"
  top: "conv4"
}
layer {
  name: "conv5"
  type: "Convolution"
  bottom: "conv4"
  top: "conv5"
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    group: 2
  }
}
layer {
  name: "relu5"
  type: "ReLU"
  bottom: "conv5"
  top: "conv5"
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "conv5"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "fc6" # 全连接层，实际也是一种卷积层只不过尺寸是1 * 1
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  inner_product_param {
    num_output: 4096 # 卷积核的个数
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
layer {
  name: "drop6"
  type: "Dropout"
  bottom: "fc6"
  top: "fc6"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc7"
  type: "InnerProduct"
  bottom: "fc6"
  top: "fc7"
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
layer {
  name: "drop7"
  type: "Dropout"
  bottom: "fc7"
  top: "fc7"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc8"
  type: "InnerProduct"
  bottom: "fc7"
  top: "fc8"
  inner_product_param {
    num_output: 1000
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc8"
  top: "prob"
}


