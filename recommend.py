import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
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


# 设定Node2Vec的参数
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

# 训练模型
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# 获取节点嵌入
embeddings = model.wv


# 获取所有动画作品的嵌入
titles = [node for node in G.nodes if G.nodes[node]['node_type'] == 'title']
title_embeddings = [embeddings[G.nodes[title]['index']] for title in titles]  # 注意：这里需要确保嵌入索引与节点匹配

# 计算相似度矩阵
similarity_matrix = cosine_similarity(title_embeddings)


# 为给定的动画作品推荐相似的作品
def recommend_titles(title, num_recommendations=5):
    title_index = titles.index(title)
    similarities = similarity_matrix[title_index]
    recommended_indices = similarities.argsort()[::-1][1:num_recommendations + 1]  # 排除自身
    return [titles[i] for i in recommended_indices]


# 示例：为某个动画作品推荐
recommended_titles = recommend_titles('某动画作品标题', num_recommendations=5)
print(recommended_titles)