# 实验三：图像分类与经典CNN实现

本次实验全部材料均在此文件夹中。

### 数据集

MNIST手写数据集均以解压并放在 `/dataset` 文件夹中。

### 代码

* `main.py` : 运行本次实验主函数
* `LeNet.py` : LeNet 网络结构实现
* `AlexNet.py` : AlexNet 网络结构实现
* `ResNet.py` : ResNet18，ResNet34，ResNet50 网络结构实现
* `DenseNet.py` : DenseNet121，DenseNet169 网络结构实现
* `VGG.py` : VGG11网络结构实现

### 运行环境依赖

运行代码所需要库在 `requirements.txt` 中，输入 `pip install -r requirements.txt` 即可完成安装。

### 运行代码

在terminal中输入类似于下面的命令，可自行决定超参数值并训练模型：

```shell
python main.py --model lenet --lr 0.02 --dropout 0.1
```

### 实验报告

具体实验思路及结果分析在 `实验报告.pdf` 中。
