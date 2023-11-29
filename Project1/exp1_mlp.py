import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import re
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


# 打开train.txt文件以读取内容
with open('./exp1_data/train_data.txt', 'r') as file:
    lines = file.readlines()

labels = []
raw_texts = []

for line in lines:
    data = eval(line)
    label = data['label']
    raw_text = data['raw']
    labels.append(label)
    raw_texts.append(raw_text)

# 小写，去除停用词和标点
stopword = set(stopwords.words('english'))
texts = []
for i in range(len(raw_texts)):
    text = raw_texts[i].lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stopword]
    text = ' '.join(filtered_words)
    texts.append(text)

# 使用TF-IDF表示文本
vectorizer = TfidfVectorizer()
text_vec = vectorizer.fit_transform(texts)
text_mat = text_vec.toarray()


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# 定义超参数
input_dim = text_mat.shape[1]
hidden_dim = 64
output_dim = len(set(labels))  # 输出维度等于类别数

# 初始化K折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=1000)

best_model = None
best_accuracy = 0.0

for train_index, valid_index in kf.split(text_mat):
    x_train, x_valid = text_mat[train_index], text_mat[valid_index]
    y_train, y_valid = np.array(labels)[train_index], np.array(labels)[valid_index]

    # 转换为PyTorch张量
    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_valid = torch.FloatTensor(x_valid)
    y_valid = torch.LongTensor(y_valid)

    # 初始化MLP模型
    model = MLPClassifier(input_dim, hidden_dim, output_dim)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 使用指数连续衰减学习率策略
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # 设置衰减因子gamma

    # 训练模型
    for epoch in range(100):  # 迭代100次
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率

    # 在验证集上评估模型
    with torch.no_grad():
        outputs = model(x_valid)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_valid).sum().item() / len(y_valid)

    # 更新最佳模型和准确率
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        precision = precision_score(y_valid, predicted, average='weighted')
        recall = recall_score(y_valid, predicted, average='weighted')
        f1 = f1_score(y_valid, predicted, average='weighted')
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
