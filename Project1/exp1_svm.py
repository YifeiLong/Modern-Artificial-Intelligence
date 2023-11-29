import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import re
from nltk.corpus import stopwords
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec


# 打开train.txt文件以读取内容
with open('./exp1_data/train_data.txt', 'r') as file:
    lines = file.readlines()

labels = []
raw_texts = []

for line in lines:
    # 使用eval函数解析JSON格式的文本行
    data = eval(line)
    label = data['label']
    raw_text = data['raw']
    labels.append(label)
    raw_texts.append(raw_text)

# 小写，去除停用词和标点
stopword = set(stopwords.words('english')) # nltk中stopword
texts = []
for i in range(len(raw_texts)):
    text = raw_texts[i].lower()
    text = re.sub(r'[^\w\s]', '', text) # 去掉标点
    # 移除停用词
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
text_vec = vectorizer.fit_transform(texts)
train_vec = vectorizer.transform(train_text)
test_vec = vectorizer.transform(test_text)

# 使用Word2Vec表示文本
# model = Word2Vec(train_text, vector_size=10000, window=5, min_count=1, sg=10)
#
# def text_to_vec(text, model):
#     return [model.wv[word] for word in text if word in model.wv]
#
# train_vec = np.zeros((len(train_text), model.vector_size))
# test_vec = np.zeros((len(test_text), model.vector_size))
#
# for i, text in enumerate(train_text):
#     vectors = [model.wv[word] for word in text if word in model.wv]
#     if vectors:
#         train_vec[i] = np.mean(vectors, axis=0)  # 使用平均Word2Vec向量
#     else:
#         train_vec[i] = np.zeros(model.vector_size)  # 如果文本中没有词在模型中出现，则返回零向量
#
# for i, text in enumerate(test_text):
#     vectors = [model.wv[word] for word in text if word in model.wv]
#     if vectors:
#         test_vec[i] = np.mean(vectors, axis=0)
#     else:
#         test_vec[i] = np.zeros(model.vector_size)

# svm
# 10折交叉验证，调参
tol_list = [1e-3, 1e-4, 1e-5]
iter_list = [100, 250, 500, 1000, 1500]
C = [0.1, 0.3, 0.5, 1, 2]

kf = KFold(n_splits=10, shuffle=True, random_state=2023)
best_model = None
best_accuracy = 0.0

for tol1 in tol_list:
    for iter_num in iter_list:
        for c1 in C:
            for train_index, valid_index in kf.split(text_vec):
                x_train, x_valid = text_vec[train_index], text_vec[valid_index]
                y_train, y_valid = np.array(labels)[train_index], np.array(labels)[valid_index]
                svm_classifier = SVC(kernel='linear', random_state=2023, C=c1, max_iter=iter_num, tol=tol1)
                svm_classifier.fit(x_train, y_train)  # 训练模型
                predictions = svm_classifier.predict(x_valid)
                accuracy = accuracy_score(y_valid, predictions)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = svm_classifier
                    print("Accuracy: ", accuracy)
                    print("C: ", c1)
                    print("tol: ", tol1)
                    print("max_iter: ", iter_num, '\n')


# 固定8-2划分，训练模型
svm_classifier = SVC(kernel='linear', random_state=2023, max_iter=1000, C=1, tol=1e-3)
svm_classifier.fit(train_vec, train_label)  # 训练模型
# 验证集上测试
predictions = svm_classifier.predict(test_vec)
accuracy = accuracy_score(test_label, predictions)
precision = precision_score(test_label, predictions, average='weighted')
recall = recall_score(test_label, predictions, average='weighted')
f1 = f1_score(test_label, predictions, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# 使用最佳模型输出预测结果
with open('./exp1_data/test.txt', 'r') as file:
    lines = file.readlines()

pred_text = []
pred_id = []
for line in lines:
    # 找到第一个逗号的位置
    first_comma_index = line.find(',')
    if first_comma_index != -1:
        id_str = line[:first_comma_index].strip()
        text = line[first_comma_index + 1:].strip()
        # 尝试将id解析为整数，如果失败则忽略该行
        try:
            id = int(id_str)
            pred_id.append(id)
            pred_text.append(text)
        except ValueError:
            print(f"Skipping line: {line}")


pred_texts = []
for i in range(len(pred_text)):
    text = pred_text[i].lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stopword]
    text = ' '.join(filtered_words)
    pred_texts.append(text)

pred_vec = vectorizer.transform(pred_texts)
res = best_model.predict(pred_vec)

pred_id = np.array(pred_id)
output_file = "results.txt"
# 写入数据
with open(output_file, "w") as file:
    # 写入标题行
    file.write("id, pred\n")
    for id, pred in zip(pred_id, res):
        file.write(f"{id}, {pred}\n")
