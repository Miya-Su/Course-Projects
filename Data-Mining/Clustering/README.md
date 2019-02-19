# Clustering



report.pdf文件:   为实验报告具体文档

clustering.ipynb文件：为实验代码



## 实验说明

实验一共分为如下四部分

#### **1.**    **数据集及预处理**

- **数据集**

本实验采用的数据来自于UCI,下载地址如下：<http://archive.ics.uci.edu/ml/datasets/Adult>

这是一组“人口普查收入”数据集，根据人口普查数据预测人们的收入是否超过5万美元/年。这组数据由Barry Becker从1994年的人口普查数据库中提取。

该数据集一共有14个属性，预测任务是确定一个人的年收入是否超过50K。

- ##### **数据处理**

  - 缺失值处理

  - 去重处理

  - One-Hot编码（标签数值化）

  - 对数据列进行处理

  - 异常值处理

  - 数据归一化处理

  - 特征选择

  - 降维处理显示

    由于原来数据集是一个具有14个特征的高维数据集，没有办法直接可视化结果，因此我们采用PCA降维的方法将原数据降低到2维，画出聚类后的二维数据图像



#### **2.**    **模型构建**

- ##### ****kmeans**** 

- ##### ****Meanshift**** 

- ##### ****MiniBirchKmeans****

  




#### **3.模型评估与分析**

- ##### CH评估指标

  Calinski-Harabasz(CH)指标通过类内离差矩阵描述紧密度，类间离差矩阵描述分离度



#### 4. 实验小结

 本实验依然采用美国居民收入统计数据集，希望通过分析居民的各个特征将居民进行分类，原数据集特征一共有14个，我们根据随机森林计算出各个特征的重要程度，进行特征筛选，删除掉影响可以忽略的特征，最终留下这八个特征：fnlwg、age、capital-gain、relationship、education-num、hours-per-week、marital-status、occupation，通过对这八个特征进行聚类分析，采用三种聚类方法：Kmenas、Meanshift、MiniBirchKmeans来进行运算，最终通过可视化图像观察以及CH（Calinski-Harabasz）指标来衡量不同聚类方法的效果，得出，Kmenas在聚类数目为三是能达到较好的聚类效果，其次是MiniBirchKmeans，但是由于MiniBirchKmeans每次都是进行抽样计算，因此每次计算的结果可能出现差异，但是在多次计算的比较中，可以发现该值均是大于Meanshift但同时又是低于Kmeans的，因此效果一般，最后Meanshift的聚类效果较差。