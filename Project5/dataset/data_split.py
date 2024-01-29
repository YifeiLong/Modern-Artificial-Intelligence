import pandas as pd
import shutil
from sklearn.utils import shuffle as reset


def train_test_split(data, test_size=0.1, shuffle=True, random_state=2024):
    if shuffle:
        data = reset(data, random_state=random_state)
    train = data[int(len(data) * test_size):].reset_index(drop=True)
    test = data[:int(len(data) * test_size)].reset_index(drop=True)
    return train, test


# 划分数据集并保存
def dataset_prepare():
    # 测试集
    test_label = pd.read_csv('test_without_label.txt', encoding='utf-8')
    test_num = test_label['guid'].copy().values
    for num in test_num:
        img_path = './dataset/data/' + str(num) + '.jpg'
        img_dest = './dataset/test/img/' + str(num) + '.jpg'
        text_path = './dataset/data/' + str(num) + '.txt'
        text_dest = './dataset/test/text/' + str(num) + '.txt'
        shutil.copy(img_path, img_dest)
        shutil.copy(text_path, text_dest)

    # 分割训练集和验证集
    raw_label = pd.read_csv('train.txt', encoding='utf-8')
    train_label, val_label = train_test_split(raw_label)

    train_label.sort_values(by="guid", inplace=True, ascending=True)
    val_label.sort_values(by="guid", inplace=True, ascending=True)
    train_label.to_csv('./dataset/train/train.csv', index=False)
    val_label.to_csv('./dataset/val/val.csv', index=False)

    train_num = train_label['guid'].copy().values
    for num in train_num:
        img_path = './data/' + str(num) + '.jpg'
        img_dest = './train/img/' + str(num) + '.jpg'
        text_path = './data/' + str(num) + '.txt'
        text_dest = './train/text/' + str(num) + '.txt'
        shutil.copy(img_path, img_dest)
        shutil.copy(text_path, text_dest)

    val_num = val_label['guid'].copy().values
    for num in val_num:
        img_path = './data/' + str(num) + '.jpg'
        img_dest = './val/img/' + str(num) + '.jpg'
        text_path = './data/' + str(num) + '.txt'
        text_dest = './val/text/' + str(num) + '.txt'
        shutil.copy(img_path, img_dest)
        shutil.copy(text_path, text_dest)


if __name__ == '__main__':
    dataset_prepare()
