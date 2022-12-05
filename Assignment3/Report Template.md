# Experiments on Bag of Visual Words for Scene Recognition

# Computer Vision Assignment 3 Report

Student Name: 叶璨铭

Student ID: 12011404

## 基础代码实现与现象观察

在开始实验调参之前，我们首先将代码填充完，将整个实验流程跑通。在这个过程中观察到一些现象，作为调参实验的设计依据。

首先实现tiny_images，初始的的准确率是15%. 接下来对feature做标准化

```python
feat = (feat - np.mean(feat)) / np.std(feat)
```

可以发现准确率显著提高7%.

![image-20221205192530741](Report Template.assets/image-20221205192530741.png)

注意观察多线程实现前后的时间运行差异，可以看出显著的加速。

使用词袋模型，用SVM前是52%, 使用SVM后显著提高了7%。

![image-20221205193112076](Report Template.assets/image-20221205193112076.png)

## Experimental: 基于交叉验证探究词表大小与性能的关系。

### Design

- 交叉验证
  - 交叉验证的目的是为了防止测试数据的信息被训练过程用到。我们调整超参数不能根据测试集的数据上的准确率，这样有利用测试集信息的嫌疑。
  - 所以在训练集上获得验证集，然后调整超参数。
- 

### Results and Analysis



## Bonus Report (If you have done any bonus problem, state them here)

### 优化：特征提取好慢，如何舒缓跑不出来的焦虑？

#### 多线程加速计算

#### tqdm显示进度

#### 使用GPU加速

我们知道`PyTorch`的本质与其说是深度学习框架，不如说是先支持了类似numpy API的GPU计算，然后附带一个自动微分库用来求导，以便构建神经网络。`sklearn`和`vlfeat`当然是不支持GPU加速的，但是我们传递给它们的都是numpy张量，既然`PyTorch`支持numpy的几乎所有API，我们能不能“偷梁换柱”一下，把numpy调用全部换成GPU加速的运算呢？

- 理论分析：sift计算过程首先卷积、降采样生成了大量的高斯金字塔，然后卷积求出梯度，然后求出极坐标，然后解方程找真正位置，然后找窗口和histogram给出特征。以上操作中有很多GPU友好的操作，比如卷积。
- 实际操作：

首先我们使用

```python
import torch as np
np.array = lambda x: np.Tensor(x).to('cuda')
image_gpu = np.array(image)
```

e完全没有问题，接下来在这个Python文件当中我们的产生的张量就是GPU上的了

首先对`OpenCV`的SIFT试试：

```python
%timeit kp, des = sift.detectAndCompute(image_gpu,None)
```


![image-20221205021020276](Report Template.assets/image-20221205021020276.png)

`OpenCV`居然用类型检查发现了不是numpy就报错了，离谱，这不python，python就是弱类型的啊！

用vlfeat试试

```python
%timeit frames, descriptors = vlfeat.sift.dsift(image_gpu, step=5, fast=True)
```

只见

```error
...
image = np.require(image, dtype=np.float32, requirements='C')
...
# Wrap Numpy array again in a suitable tensor when done, to support e.g.
...
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```

仔细看报错信息，原来vlfeat 库也是先把传入进来的image先变成了numpy的数组。或者说它制造了一个返回值，要求image变成umpy array是这样的。



#### 使用更快的sift实现

![image-20221205012336037](Report Template.assets/image-20221205012336037.png)

![image-20221205012326985](Report Template.assets/image-20221205012326985.png)

使用opencv来提取sift特征，比cyvlfeat稍微快70ms左右。

但是这两个算法的参数空间是不一样的，没有可比性，不能确定效果差异怎么样，因为控制方法不一样。

### 实验：最优参数演化计算



### 实验：朴素贝叶斯会不会更好？



