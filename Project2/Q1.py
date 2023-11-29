import numpy as np
import queue


# 寻找空格位置
def find_space(data_now):
    x0, y0 = np.where(data_now == 0)
    return x0[0], y0[0]


# 移动空格位置
# data_now: 当前网格中节点布局数组
def swap(data_now, direction):
    x, y = find_space(data_now)  # 寻找空格位置
    pos = np.copy(data_now)
    if direction == 'left':
        # 最左边，不能左移
        if y == 0:
            return pos
        else:
            # 左移更新位置
            pos[x][y] = pos[x][y - 1]
            pos[x][y - 1] = 0
            return pos
    elif direction == 'right':
        if y == 2:
            return pos
        else:
            pos[x][y] = pos[x][y + 1]
            pos[x][y + 1] = 0
            return pos
    elif direction == 'up':
        if x == 0:
            return pos
        else:
            pos[x][y] = pos[x - 1][y]
            pos[x - 1][y] = 0
            return pos
    elif direction == 'down':
        if x == 2:
            return pos
        else:
            pos[x][y] = pos[x + 1][y]
            pos[x + 1][y] = 0
            return pos


# 启发式函数，当前状态距离目标状态之间还相差多少个格子不同
def h(pos, end_data):
    total = 0
    for i in range(3):
        for j in range(3):
            if pos[i][j] != end_data[i][j]:
                total += 1
    return total


# 将当前9宫格状态转化为一个序列，方便作为closed的key
def data_to_int(num):
    val = 0
    for i in range(3):
        for j in range(3):
            val = val * 10 + num[i][j]
    return val


# 加入队列中的每个节点信息
class Node:
    f = -1  # f函数值
    step = 0  # 初始状态到当前状态的步数
    data = None  # 当前的状态（9宫格）

    def __init__(self, data, step, end_data):
        self.data = data
        self.step = step
        self.f = h(data, end_data) + step


# 给open队列中元素排序
def sort_by_f(opened):
    open1 = opened.queue.copy()
    l = len(open1)
    for i in range(l):
        for j in range(l):
            if open1[i].f < open1[j].f:
                t = open1[i]
                open1[i] = open1[j]
                open1[j] = t
            elif open1[i].f == open1[j].f:
                if open1[i].step > open1[j].step:
                    t = open1[i]
                    open1[i] = open1[j]
                    open1[j] = t
    opened.queue = open1
    return opened


# 判断当前插入节点在opened队列中是否出现过，如果出现过，就保留f值更小的节点
def update_open(node, opened):
    open1 = opened.queue.copy()
    for i in range(len(open1)):
        data = open1[i]
        if (data == node.data).all():
            if open1[i].f <= node.f:
                return opened
            else:
                open1[i] = node
                opened.queue = open1
                return opened

    open1.append(node)
    opened.queue = open1
    return opened


# 算法主函数
def eight_digit(opened, end_data):
    move = 0  # 总移动次数
    while len(opened.queue) != 0:
        node = opened.get()
        # 达到终止状态
        if (node.data == end_data).all():
            print(move)
            return

        # 将取出的点加入closed中
        closed[data_to_int(node.data)] = 1

        for action in ['left', 'right', 'up', 'down']:
            child_node = Node(swap(node.data, action), node.step + 1, end_data)
            index = data_to_int(child_node.data)
            if index not in closed:
                opened = update_open(child_node, opened)

        opened = sort_by_f(opened)
        move += 1


if __name__ == '__main__':
    input_str = input()
    nums = [int(char) for char in input_str]
    start_data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            start_data[i][j] = nums[i * 3 + j]
    start_data = np.array(start_data)
    end_data = np.array([[1, 3, 5], [7, 0, 2], [6, 8, 4]])
    opened = queue.Queue()
    start_node = Node(start_data, 0, end_data)
    opened.put(start_node)
    # 遍历过的节点，用字典方便查找
    closed = {}
    eight_digit(opened, end_data)
