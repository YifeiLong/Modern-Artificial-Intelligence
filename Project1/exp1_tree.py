import numpy as np
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from nltk.corpus import stopwords


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

# 使用TF-IDF表示文本
vectorizer = TfidfVectorizer()
text_mat = vectorizer.fit_transform(texts)

# k折交叉验证
# 将样本等分为10份，每次抽取一份作为验证集
kf = KFold(n_splits=10, shuffle=True, random_state=2023)
# train, valid都是对应行号的数组
best_model = None
best_accuracy = 0.0

for train_index, valid_index in kf.split(text_mat):
    x_train, x_valid = text_mat[train_index], text_mat[valid_index]
    y_train, y_valid = np.array(labels)[train_index], np.array(labels)[valid_index]

    # 训练决策树模型
    tree = DecisionTreeClassifier(random_state=10, criterion="entropy", max_depth=3)
    tree.fit(x_train, y_train)

    # 在验证集上评估模型
    accuracy = tree.score(x_valid, y_valid)

    # 更新最佳模型和准确率
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = tree

# 输出最佳模型的验证集准确率
print("Best accuracy:", best_accuracy)
