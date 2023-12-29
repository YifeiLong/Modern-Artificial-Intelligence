# 实验四：文本摘要

### 环境配置

在命令行中输入下面的命令即可完成环境配置：

```shell
pip install -r requirements.txt
```

**注：**GRU模型最好使用Google Colab配置好相关包后运行，在本地运行可能会产生错误。



### 代码运行

#### \- 逐个执行

如果想要运行GRU模型，则在命令行中输入（参数也可以省略）：

```sh
python GRU.py --embed_size 16 --lr 0.01 --hidden_size 16 --dropout 0.1
```

如果想要运行BART模型，则在命令行中输入（参数也可以省略）：

```sh
python BART.py --lr 0.01 --epoch 20 --batch_size 50
```

#### \- 统一执行

需要使用脚本运行，首先在命令行中输入以下命令，赋予脚本执行权限：

```sh
chmod +x run.sh
```

然后在终端输入以下内容运行脚本：

```sh
./run.sh
```



### 文件内容

* `/data`：存放训练所用数据集
* `/model`：保存了GRU和BART模型训练得到的最好模型（.pth文件）
* `BART.py`：BART模型训练与测试代码
* `GRU.py`：GRU模型训练与测试代码
* `requirements.txt`：运行代码所需要配置的依赖
* `run.sh`：运行代码脚本，可以运行该脚本，也可以按照上面的方式单个代码运行
* 实验报告pdf
