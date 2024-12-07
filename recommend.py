import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec
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



# 提取动画作品节点
title_nodes = [node for node in G.nodes if G.nodes[node]['node_type'] == 'title']
# 初始化 Node2Vec 模型，设置合适的参数（可根据实际情况调整）
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
# 训练模型得到动画作品节点的向量表示
model = node2vec.fit(window=10, min_count=1, batch_words=4)
# 获取所有动画作品节点的向量表示
title_vectors = {title: model.wv[title] for title in title_nodes}
print("图嵌入结束")
# 假设要为动画作品 'title1' 进行推荐（需确保 'title1' 存在于图中）
def recom(input_title):
    if input_title in title_nodes:
        target_vector = title_vectors[input_title].reshape(1, -1)
        similarities = cosine_similarity(target_vector, list(title_vectors.values()))[0]
        # 获取相似度排名靠前的其他动画作品
        sorted_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[1:10]  # 去掉自身（相似度为 1）
        recommended_titles = [title_nodes[i] for i, _ in sorted_similarities]
        # 按照rating和num_rating属性进行排序
        sorted_recommended_titles = sorted(recommended_titles,
                                           key=lambda x: (-G.nodes[x]['rating'], -G.nodes[x]['num_ratings']))
        print(f"为动画作品 {input_title} 推荐的其他动画作品（按照评分和评分数量排序）：")
        for input_title in sorted_recommended_titles:
            print(f"动画作品：{input_title}，评分：{G.nodes[input_title]['rating']}，评分数量：{G.nodes[input_title]['num_ratings']}")
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

recom("鬼灭之刃")








# import community as community_louvain
#
# # 计算最佳分区（社区划分）
# partition = community_louvain.best_partition(G)
#
# # 为每个节点添加社区属性到图中
# for node, community_id in partition.items():
#     G.nodes[node]['community'] = community_id
#
# # 统计每个社区的节点数量
# community_sizes = {}
# for node in G.nodes():
#     community_id = G.nodes[node]['community']
#     if community_id not in community_sizes:
#         community_sizes[community_id] = 0
#     community_sizes[community_id] += 1
#
# print("社区及其对应的节点数量:", community_sizes)
#
# # 可视化社区（简单示例，用不同颜色表示不同社区，这里仅示意，实际可能需更精细调整）
# import matplotlib.pyplot as plt
# pos = nx.spring_layout(G)
# plt.figure(figsize=(10, 8))
# cmap = plt.get_cmap('viridis', max(community_sizes.keys()) + 1)
# for node in G.nodes():
#     community_id = G.nodes[node]['community']
#     nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=cmap(community_id), node_size=100)
# nx.draw_networkx_edges(G, pos, alpha=0.3)
# nx.draw_networkx_labels(G, pos, font_size=8)
# plt.title("图的社区划分可视化")
# plt.show()
