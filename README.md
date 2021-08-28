# 一点资讯技术编程大赛——CTR预估

队伍——八月无烦恼（西安交通大学，刘启东，赵成成，马黛露丝）

## 代码结构

```bash
├── data # 数据处理模块
│   ├── DA_din.ipynb # DIN数据预处理脚本
│   ├── DA_esmm.ipynb # ESMM数据预处理脚本
│   ├── handle_title.ipynb # 处理文章标题
├── examples # 实例模块
│   ├── run_DeepFM.py
│   ├── run_FM.py
│   └── run_xDeepFM.py # 运行 XDeepFM
├── generators
│   ├── generator_m.py  # 多任务ESMM的数据加载类
│   ├── generator.py # DataSet 和 Data Generator
├── log # 日志
│   ├── tensorboard
│   └── text
├── models # 模型模块
│   ├── layers # 包含输入和序列组件
│   	├── _loss.py
│   	├── input.py
│   ├── basemodel.py
│   ├── deepfm.py
│   ├── dnn.py
│   ├── esmm.py
│   ├── FM.py
│   └── xdeepfm.py
├── submission
│   ├── average.ipynb # 用于将 n 次结果平均
└── utils # 工具
    ├── candidate_generator.py
    ├── evaluation.py
    ├── selection.py
    └── utils.py
├── grid_search.py  # 对超参数进行grid search
├── main_MT.py  # 多任务的主函数入口
├── main.py # 主函数（入口）
├── README.md
├── run_din.bash # 运行DIN脚本
├── run_esmm.bash # 多任务模型运行脚本
```

## 1. xDeepFM & ESMM

### 思路

该方案以xDeepFM模型基础，构建多任务模型。利用训练数据中的观看时长和点击数据作为多任务的目标。由于观看时长是在点击之后才会产生，因此两者的关系类似于点击率和转化率的关系。故这里使用ESMM多任务模型分别预测点击率和点击后观看时长，利用观看时长任务来辅助点击任务。最终预测使用只使用点击任务的输出。

### 运行代码

1. 先将所有数据集解压后放入/data/文件夹下，然后运行DA_esmm.ipynb预处理数据。
2. 运行
```
bash run_esmm.bash
```
具体的命令行参数在main_MT.py中。
最优的一组超参数为：

```
batch_size=8192
learning_rate=0.001
epoch=1
embedding_size=64
lambda=0.01 # 训练多任务时，损失函数的加权值
```
为了使结果稳定，提交结果为5次训练模型预测结果的平均值。

## 2. XDeepFM & DIN

### 思路

该方案以 XDeepFM 为基础，融入 DIN。具体融入方式如下：将用户历史点击的文章序列（如：选取用户最近点击的15个文章）输入模型，对历史点击的文章序列做 Attention 和 Average Pooling 后，输入 XDeepFM。

### 运行

1. 先将所有数据集解压后放入/data/文件夹下，然后运行DA_din.ipynb预处理数据。
2. 运行

```
bash run_din.bash
```

具体的命令行参数在main_MT.py中。
最优的一组超参数为：

```
batch_size=8192
learning_rate=0.001
epoch=1
embedding_size=64
lambda=0.01 # 训练多任务时，损失函数的加权值
```

为了使结果稳定，提交结果为5次训练模型预测结果的平均值。

## 环境依赖
```
pip install -r requirements.txt
```

