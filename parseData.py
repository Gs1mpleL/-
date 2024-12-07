import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为一个支持中文的字体，如"SimHei"
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
# 读取CSV文件
df = pd.read_csv('data.csv')
# 示例数据清洗步骤（根据你的数据实际情况调整）
df = df.dropna(subset=['title', 'rating', 'num_ratings', 'tags','actors'])  # 删除缺失值


# 创建一个空的无向图
G = nx.Graph()
# 添加动画作品节点
for index, row in df.iterrows():
    title = row['title']
    G.add_node(title, node_type='title', rating = row['rating'], num_ratings = row['num_ratings'])

# 添加标签节点和边
for index, row in df.iterrows():
    tags = row['tags'].split('|')  # 假设标签是以'|'分隔的字符串
    for tag in tags:
        tag = tag.strip('[]').strip()  # 去除可能的方括号和空格
        if tag not in G.nodes:
            G.add_node(tag, node_type='tag')
        G.add_edge(row['title'], tag)

# 添加演员节点和边
for index, row in df.iterrows():
    actors = row['actors'].split('|')  # 假设标签是以'|'分隔的字符串
    for actor in actors:
        if "配音" in actor:
            continue
        actor = actor.strip('[]\'\" ').replace(' ', '_')  # 去除可能的方括号、引号、空格，并用下划线替换空格
        if actor not in G.nodes:
            G.add_node(actor, node_type='actor')
        G.add_edge(row['title'], actor)


def show():
    # 可视化图
    pos = nx.spring_layout(G)  # 使用spring布局算法
    # 设置节点颜色
    node_colors = [G.nodes[node]['node_type'] for node in G.nodes()]
    color_map = ['red' if type_ == 'title' else ('blue' if type_ == 'tag' else 'green') for type_ in node_colors]
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=5, node_color=color_map, cmap=plt.cm.rainbow)
    # 绘制边
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), alpha=0.1)
    # 绘制标签
    # nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    # 显示图形
    plt.title('Animation Graph')
    plt.text(0, 0, '红色：动画\n蓝色：标签\n绿色：配音演员', fontsize=12, color='black', transform=plt.gca().transAxes, ha='left', va='bottom')
    plt.show()


# 计算动画之间的相似性
def compute_similarity(input_G, title1, title2):
    # 获取两个动画的邻居（标签和演员）
    neighbors1 = set(input_G.neighbors(title1))
    neighbors2 = set(input_G.neighbors(title2))

    # 计算共享的邻居数量
    common_neighbors = neighbors1.intersection(neighbors2)
    similarity = len(common_neighbors)

    # 也可以考虑加入权重，比如根据标签或演员的重要性
    # 这里我们简单地使用数量作为相似性度量

    return similarity


# 根据相似性进行推荐
def recommend_animations(G, input_title, num_recommendations=5):
    # 获取输入动画的节点信息
    input_node = input_title
    if input_node not in G.nodes:
        raise ValueError(f"Animation '{input_title}' not found in the graph.")

    # 计算所有其他动画与输入动画的相似性
    similarities = {}
    for title in G.nodes:
        if title != input_node and G.nodes[title]['node_type'] == 'title':
            sim = compute_similarity(G, input_node, title)
            similarities[title] = sim

    # 根据相似性和评分对动画进行排序
    # 我们先按相似性排序，如果相似性相同，则按rating降序，再按num_ratings降序
    sorted_animations = sorted(similarities.items(), key=lambda item: (
    -item[1], -G.nodes[item[0]]['rating'], -G.nodes[item[0]]['num_ratings']))

    # 提取前num_recommendations个动画
    recommendations = sorted_animations[:num_recommendations]

    # 准备推荐结果
    rec_list = []
    for title, sim in recommendations:
        rec_dict = {
            'title': title,
            'similarity': sim,
            'rating': G.nodes[title]['rating'],
            'num_ratings': G.nodes[title]['num_ratings']
        }
        rec_list.append(rec_dict)

    return rec_list


# 示例使用
input_animation = '鬼灭之刃'
recommendations = recommend_animations(G, input_animation)
for rec in recommendations:
    print(
        f"Recommended Animation: {rec['title']}, Similarity: {rec['similarity']}, Rating: {rec['rating']}, Num Ratings: {rec['num_ratings']}")
show()