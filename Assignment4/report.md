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
  - 如何调整滑动窗口的参数？如何调整正例阈值？



## Experimental: 基于交叉验证探究词表大小与性能的关系。




## Bonus Report (If you have done any bonus problem, state them here)



### 实验：最优参数演化计算

### 实验：朴素贝叶斯会不会更好？



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





