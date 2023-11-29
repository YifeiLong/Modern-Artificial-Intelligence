import queue
import numpy as np


class Node:
    index = 0  # 当前节点楼层号
    g = 0  # 从起点开始的实际路径长
    h = 0  # 启发式函数值

    def __init__(self, index, g, h):
        self.index = index
        self.g = g
        self.h = h

    # 优先级队列排序方式，g+h值小的优先级高
    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)


# Dijkstra求解到终点最短距离，作为启发式函数
def dijkstra(graph, start, end):
    open = {}  # 起点
    close = {}  # 终点
    open[start] = 0  # 起点放进表中

    while open:
        # 取出距离最小的节点
        distance, min_node = min(zip(open.values(), open.keys()))
        open.pop(min_node)
        # 将节点加入close
        close[min_node] = distance
        # 找到终点
        if min_node == end:
            return distance

        # 遍历当前节点的邻居节点
        for node in graph.get(min_node, {}).keys():
            # 邻居节点未被展开过
            if node not in close.keys():
                # 邻居节点在open中
                if node in open.keys():
                    # 如果当前新距离小于原来的情况，更新open中数值
                    if graph[min_node][node] + distance < open[node]:
                        open[node] = graph[min_node][node] + distance
                # 邻居节点不在open中
                else:
                    open[node] = graph[min_node][node] + distance

    # 两点之间没有路径，启发式函数值返回1
    if end not in close:
        return 1


def a_star(graph, reverse_graph, K, start, end):
    open = queue.PriorityQueue()
    open.put(Node(start, 0, dijkstra(reverse_graph, end, 1)))
    time = 0  # 找到的最短路径数量
    visit = np.zeros(end)  # 当前编号节点被访问了几次
    res_dis = np.full(K, -1)

    while not open.empty():
        # 取出当前open队首节点
        node_now = open.get()
        id = node_now.index - 1
        visit[id] += 1  # 访问次数+1
        # 该节点被展开过K次，不再展开
        if visit[id] > K:
            continue
        if node_now.index == end:
            res_dis[time] = node_now.g  # 第time个结果
            time += 1
        # 已经有K个结果
        if visit[end - 1] == K:
            return res_dis

        # 将邻居节点加入队列
        neighbors = graph.get(node_now.index, {}).keys()
        for neighbor in neighbors:
            open.put(Node(neighbor, (node_now.g + graph[node_now.index][neighbor]),
                          dijkstra(reverse_graph, end, neighbor)))

    return res_dis


if __name__ == '__main__':
    graph = {}
    reverse_graph = {}
    N, M, K = map(int, input().split())
    for i in range(M):
        start, end, cost = map(int, input().split())
        if start < end:
            if start not in graph:
                graph[start] = {}
            graph[start][end] = cost
            if end not in reverse_graph:
                reverse_graph[end] = {}
            reverse_graph[end][start] = cost

    res = a_star(graph, reverse_graph, K, 1, N)
    for item in res:
        print(item)
