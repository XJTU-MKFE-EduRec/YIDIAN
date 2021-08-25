[toc]

# 一点资讯 CTR -- XDeepFM & DIN

## 思路

该方案以 XDeepFM 为基础，融入 DIN。具体融入方式如下：将用户历史点击的文章序列（如：选取用户最近点击的15个文章）输入模型，对历史点击的文章序列做 Attention 和 Average Pooling 后，输入 XDeepFM。

## 代码结构

```bash
.
├── data # 数据处理模块
│   ├── DA.ipynb # 数据预处理脚本
│   ├── handle_title.ipynb
├── examples # 实例模块
│   ├── run_DeepFM.py
│   ├── run_FM.py
│   └── run_xDeepFM.py # 运行 XDeepFM
├── generators
│   ├── generator_m.py
│   ├── generator.py # DataSet 和 Data Generator
├── grid_search.py
├── log # 日志
│   ├── tensorboard
│   └── text
├── main_MT.py
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

## 依赖

- python
- pytorch
- numpy
- pandas
- tqdm
- setproctitle
- tensorboard

