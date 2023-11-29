import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
word_list = open("stopwords.txt", 'r').read()
word_list = set(word_list)
stopword = stopword | word_list

texts = []
for i in range(len(raw_texts)):
    text = raw_texts[i].lower()
    text = re.sub(r'[^\w\s]', '', text) # 去掉标点
    words = text.split()
    filtered_words = [word for word in words if word not in stopword]
    text = ' '.join(filtered_words)
    texts.append(text)

# 划分训练集、验证集（8:2）
# 每个类别800个样本，共10类，验证集160个
train_text = []
test_text = []
train_label = []
test_label = []

for i in range(10):
    list1 = random.sample(range(800 * i, 800 * (i + 1)), 160)
    for j in list1:
        test_text.append(texts[j])
        test_label.append(i)
    t = set(range(800 * i, 800 * (i + 1)))
    list2 = list(t.difference(set(list1))) # 取差集
    for k in list2:
        train_text.append(texts[k])
        train_label.append(i)

# 使用TF-IDF表示文本
vectorizer = TfidfVectorizer()
train_vec = vectorizer.fit_transform(train_text)
test_vec = vectorizer.transform(test_text)

x_train = train_vec.toarray()
y_train = np.array(train_label)
y_train = y_train.transpose()

x_test = test_vec.toarray()
y_test = np.array(test_label)
y_test = y_test.transpose()


class LogisticRegression:
    def __init__(self, learning_rate, iteration, num_classes):
        self.theta = None
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.num_classes = num_classes

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def one_hot_encode(self, y):
        y_encoded = np.zeros((len(y), self.num_classes))
        for i in range(len(y)):
            y_encoded[i, y[i]] = 1
        return y_encoded

    def fit(self, x, y, Lambda):
        m, n = x.shape
        self.theta = np.zeros((n, self.num_classes))

        for _ in range(self.iteration):
            z = np.dot(x, self.theta)
            h = self.softmax(z)
            gradient = np.dot(x.T, (h - self.one_hot_encode(y))) / m
            self.theta -= self.learning_rate * (gradient + 2 * Lambda * self.theta) # 正则化

    def predict(self, x):
        z = np.dot(x, self.theta)
        h = self.softmax(z)
        return np.argmax(h, axis=1) # 返回概率值最大分类


model = LogisticRegression(learning_rate=0.01, iteration=1000, num_classes=10)
model.fit(x_train, y_train, Lambda=0.01)
predictions = model.predict(x_test)

accuracy = accuracy_score(test_label, predictions)
precision = precision_score(test_label, predictions, average='weighted')
recall = recall_score(test_label, predictions, average='weighted')
f1 = f1_score(test_label, predictions, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
