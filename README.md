# 一点资讯CTR--xDeepFM & ESMM

## 思路
该方案以xDeepFM模型基础，构建多任务模型。利用训练数据中的观看时长和点击数据作为多任务的目标。由于观看时长是在点击之后才会产生，因此两者的关系类似于点击率和转化率的关系。故这里使用ESMM多任务模型分别预测点击率和点击后观看时长，利用观看时长任务来辅助点击任务。最终预测使用只使用点击任务的输出。

## 代码结构
```bash
├── data # 数据处理模块
│   ├── DA.ipynb # 数据预处理脚本
│   ├── handle_title.ipynb
├── examples # 实例模块
│   ├── run_DeepFM.py
│   ├── run_FM.py
│   └── run_xDeepFM.py # 运行 XDeepFM
├── generators
│   ├── generator_m.py  # 多任务ESMM的数据加载类
│   ├── generator.py # DataSet 和 Data Generator
├── grid_search.py  # 对超参数进行grid search
├── log # 日志
│   ├── tensorboard
│   └── text
├── main_MT.py  # 多任务的主函数入口
├── main.py # 主函数（入口）
├── models # 模型模块
│   ├── basemodel.py
│   ├── deepfm.py
│   ├── dnn.py
│   ├── esmm.py
│   ├── FM.py
│   ├── layers # 包含输入和序列组件
│   └── xdeepfm.py  # XDeepFM 模型
├── README.md
├── run.bash # 运行脚本
├── submission
│   ├── average.ipynb # 平均脚本，用于将 n 次结果平均
└── utils # 工具
    ├── candidate_generator.py
    ├── evaluation.py
    ├── selection.py
    └── utils.py
```

## 运行代码
1. 先将所有数据集解压后放入/data/文件夹下，然后运行DA.ipynb预处理数据
2. 运行
```
bash run.bash
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
1. 为了使结果稳定，提交结果为5次训练模型预测结果的平均值。

## 环境依赖
```
pip install -r requirements.txt
```
