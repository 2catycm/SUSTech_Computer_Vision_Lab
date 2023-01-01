# Experiments on Face Detection with a Sliding Window

# Computer Vision Assignment 4 Report

Student Name: 叶璨铭

Student ID: 12011404

## Project目标

上一个Assignment中，我们已经学习了如何通过词袋SIFT模型，来对整张图片识别类别。

本次Assignment的代码流程是

1. 从训练集数据（整张图片是人脸或者不是人脸）中提取HoG特征，总结人脸和非人脸的特征集合。

   - `get_positive_features`, `get_random_negative_features`

   - `mine_hard_negs`

2. 根据人脸和非人脸的特征集合，训练SVM判别器。
   - `train_classifier`

3. 完成Sliding Window的代码，在不同的尺度上对测试集数据中的人脸的位置进行检测。
   - `run_detector`

其中，有几个问题需要重点探究。

- 在第一步中，
  - 如何使用数据增强来增加特征的稳定性？
  - 如何在不同尺度下随机采样特征，增加特征的稳定性?
  - 如何调整hog特征的参数？
- 在第二步中，
  - 如何通过学习器的假阳性结果进一步训练学习器？
  - 如何调整SVM的正则项？
- 在第三步中，
  - 如何实现滑动窗口？
    - 将测试图像提取不同尺度。
    - 在尺度上转换hog特征。
    - 使用svm分类，如果超过confidence阈值，保留这个框。
  - 如何调整滑动窗口的参数？如何调整正例阈值？

## 初步实现与问题分析

使用git版本控制，我们首先获得了能够跑通全流程，经过仔细代码审查，逻辑没有问题的代码。



可以看到，

```bash
Accuracy = 99.992%
True Positive rate = 100.000%
False Positive rate = 0.030%
True Negative rate = 99.970%
False Negative rate = 0.000%
```

![image-20230101195030754](report.assets/image-20230101195030754.png)

似乎很不错，svm看起来抓住了特征，能够将脸和非脸分开。靠右的中间有一条竖线很高，很有可能是因为这些向量是都是支持向量，到间隔的距离都是零点几。

然而

![image-20230101194336846](report.assets/image-20230101194336846.png)

我们的topk是15甚至是150的时候，什么都检测不出来：

![image-20230101195509596](report.assets/image-20230101195509596.png)

我们调整topk为1500, 终于，有了一点点对的。但是其他的框无法被过滤掉，因为他们排在正确的框的前面。

![image-20230101194703602](report.assets/image-20230101194703602.png)

虽然svm自认为准确率超高，实际在滑动窗口的过程中对人脸的认知完全是错误的，把不是人脸的对象排序了在人脸的前面很多。

我们可以排除这是因为svm没有交叉验证，写一个交叉验证，可以看到svm对已有的data中的特征识别完全正确：

​	![image-20230101195006104](report.assets/image-20230101195006104.png)



### 可能遇到的逻辑问题

- 项目要求的xy方向与图像的行列方向不一致。这是一个很坑的问题。
- 多线程执行器忽略了异常。Python作为弱类型语言，很多问题在这里都被忽略了，比如数据类型不是int引发的崩溃，参数传递名字不对引发的崩溃，参数看起来传递成功实际上顺序被偷换导致的崩溃。
- 

### Debug方法

1. 灵活使用joblib

   1. 对于已经确定没有bug的代码，使用joblib可以跳过他们的正确运算，加快速度，瞬间到达有问题的代码位置
   2. 对于可能有bug，正在调整的代码，不能使用joblib

2. 灵活使用多线程。

   1. 对于无bug代码，多线程显著提高速度。
   2. 对于有锁的多线程（通常因为返回值长度不确定而必须有竞争关系），谨慎使用
   3. 对于有bug代码，去除多线程。

3. 由于cyvlfeat是个破旧的库，死活不支持python3.6以上哪怕是3.7，而vscode又早就宣布放弃python3.6, vscode无法debug。必须使用Pycharm，而且不能用jupyter，必须把jupyter的代码导出为普通python文件。这一点不能从conda环境上死磕，纯粹是浪费时间。

   [vscode python 3.6 无法debug 解决方案 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/541449209#:~:text=由于python开发组已停止对python3.6的维护，导致vscode当前新版本的python插件无法使用，且无法对代码debug，解决方案如下：,方案一：更换python版本为3.7及以上)

4. 修改老师的评估代码，使其支持少量的（比如10个，5个）样本评估，加快运行速度，不然debug一轮不知道猴年马月。

### 问题分析

我们首先怀疑划窗算法有问题，这个函数是最长的一个，包含了高斯金字塔的创建和单个尺度下代码的预测。

注意到在conf较大时，通常都是在octave=0时取得，而我设置的后面的这么多octave全部失效了。

我们进入debug模式，

- 将octave对齐到非0的时候，可以看到对于这张测试图片，octave=1的时候人脸的大小是对的，这个时候进行滑动窗口。

- 把i和j对齐到一定有人脸的位置，运行svm

![image-20230101181557160](report.assets/image-20230101181557160.png)

于是我们发现了问题，对于明显的人脸，svm给出的距离是-4, 是属于非人脸。这根本不是“mine hard negative”，我们的现实情况是在金字塔运行的时候，hard positive无法检测出来，而不是hard negative意外检出。

进一步推测，我们可以发现，这张照片比较模糊，这是由于我们金字塔就是逐层缩小，而原本没有那么模糊的。

![image-20230101183552530](report.assets/image-20230101183552530.png)

同样的，这张图滑动窗口没有任何问题，明明已经成功检测到这个位置，但是svm给的confidence非常离谱，最终导致战略误判。

## 代码思路的调整

经过和同学交流，发现了一个普遍规律

- 实现正确的同学不需要调参，multiscale瞬间就有0.35， 而不是0.01.
- 实现错误的同学基本上都是犯了一个错误：
  - **在原图上运行滑动窗口**，然后计算hog
- 而实现正确的同学

## Bonus Report (If you have done any bonus problem, state them here)

### 使用有意义的公式来计算金字塔的层数

```
octaves = max(
    feature_params.get('octaves', int(math.log2(min(im_shape) / win_size) / math.log2(1 / scale_factor)) + 1),
    1)  # 至少一次。
```

### 除了mine_hard_negative, 还尝试了mine_hard_positive





# 词汇表

- Image patches
  - 图像块。
- heterogeneous
  - hetero前缀：各不相同的。
  - 由很多类组成的
- crop，mirror，warp
  - 裁剪，镜像，弯曲。
- frontal faces
  - 正面的脸
- arguably
  - 可以说



# 参考文献





