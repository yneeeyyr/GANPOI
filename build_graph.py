""" Build the user-agnostic global trajectory flow map from the sequence data """
'''import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from Geohash import geohash
from param_parser import parameter_parser

import random


def add_noise_to_location(lat, lon, noise_factor=0.0001):
    """ 对POI的经纬度进行扰动,模拟位置误差 """
    lat_noise = random.uniform(-noise_factor, noise_factor)
    lon_noise = random.uniform(-noise_factor, noise_factor)
    return lat + lat_noise, lon + lon_noise

def random_edge_weight_adjustment(weight, noise_factor=0.2):
    """ 对边的权重进行扰动 """
    noise = random.uniform(-noise_factor, noise_factor)
    return max(1, weight + int(weight * noise))

def random_edge_dropout(G, dropout_rate=0.05):
    """ 随机删除图中的部分边 """
    edges = list(G.edges())
    num_edges_to_remove = int(len(edges) * dropout_rate)
    edges_to_remove = random.sample(edges, num_edges_to_remove)
    G.remove_edges_from(edges_to_remove)
    print(f"Removed {num_edges_to_remove} edges.")

def add_virtual_nodes(G, num_virtual_nodes=10):
    """ 增加虚拟节点，模拟新增POI """
    for i in range(num_virtual_nodes):
        node_id = f"virtual_{i}"
        G.add_node(node_id, checkin_cnt=0)  # 虚拟节点的签到次数设为0
        print(f"Added virtual node: {node_id}")

def build_global_POI_checkin_graph_with_augmentation(df, exclude_user=None, augment=True):
    G = nx.DiGraph()
    users = list(set(df['user_id'].to_list()))
    if exclude_user in users: users.remove(exclude_user)
    loop = tqdm(users)

    for user_id in loop:
        user_df = df[df['user_id'] == user_id]

        # Add node (POI)
        for i, row in user_df.iterrows():
            node = row['POI_id']
            lat = row['latitude']
            lon = row['longitude']
            if augment:
                # 添加位置扰动
                lat, lon = add_noise_to_location(lat, lon)

            if node not in G.nodes():
                G.add_node(node,
                           checkin_cnt=1,
                           poi_catid=row['POI_catid'],
                           poi_catid_code=row['POI_catid_code'],
                           poi_catname=row['POI_catname'],
                           latitude=lat,
                           longitude=lon)
            else:
                G.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in seq)
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['POI_id']
            traj_id = row['trajectory_id']
            # No edge for the begin of the seq or different traj
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue

            # Add edges with augmentation (random weight adjustment)
            if G.has_edge(previous_poi_id, poi_id):
                new_weight = random_edge_weight_adjustment(G.edges[previous_poi_id, poi_id]['weight'])
                G.edges[previous_poi_id, poi_id]['weight'] = new_weight
            else:
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_traj_id = traj_id
            previous_poi_id = poi_id

    if augment:
        # 可选：随机删除边、增加虚拟节点等
        random_edge_dropout(G, dropout_rate=0.05)  # 5%的边被删除
        add_virtual_nodes(G, num_virtual_nodes=5)  # 增加5个虚拟节点

    return G

def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, 'graph.pkl'), 'wb'))

def save_graph_to_csv(G, dst_dir):
    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    np.savetxt(os.path.join(dst_dir, 'graph_A.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, 'graph_X.csv'), 'w') as f:
        print('node_name/poi_id,checkin_cnt,poi_catid,poi_catid_code,poi_catname,latitude,longitude', file=f)
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1].get('checkin_cnt', 0)  # Default to 0 if missing
            poi_catid = each[1].get('poi_catid', -1)  # Default to -1 if missing (assuming integer)
            poi_catid_code = each[1].get('poi_catid_code', -1)  # Default to -1 if missing (assuming integer)
            poi_catname = each[1].get('poi_catname', 'Unknown')  # Default to 'Unknown' if missing
            latitude = each[1].get('latitude', 0.0)  # Default to 0.0 if missing (float)
            longitude = each[1].get('longitude', 0.0)  # Default to 0.0 if missing (float)
            print(f'{node_name},{checkin_cnt},'
                  f'{poi_catid},{poi_catid_code},"{poi_catname}",'
                  f'{latitude},{longitude}', file=f)
            

def save_graph_edgelist(G, dst_dir):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join(dst_dir, 'graph_node_id2idx.txt'), 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node}, {i}', file=f)

    with open(os.path.join(dst_dir, 'graph_edge.edgelist'), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[src_node]} {node_id2idx[dst_node]} {weight}', file=f)

def save_graph_with_augmentation(G, dst_dir):
    # 保存增强后的图
    save_graph_to_pickle(G, dst_dir)
    save_graph_to_csv(G, dst_dir)
    save_graph_edgelist(G, dst_dir)

# 在主程序中使用增强的构建函数
if __name__ == '__main__':
    dst_dir = r'dataset/NYC1'

    # 读取原始数据
    train_df = pd.read_csv(os.path.join(dst_dir, 'NYC_train.csv'))

    # 构建增强的全球POI签到图
    print('Build global POI checkin graph with augmentation -----------------------------------')
    G_augmented = build_global_POI_checkin_graph_with_augmentation(train_df)

    # 保存增强后的图
    save_graph_with_augmentation(G_augmented, dst_dir=dst_dir)'''


import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from Geohash import geohash
from param_parser import parameter_parser

def build_global_POI_checkin_graph(df, exclude_user=None):
    G = nx.DiGraph()
    users = list(set(df['user_id'].to_list()))
    if exclude_user in users: users.remove(exclude_user)
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user_id'] == user_id]

        # Add node (POI)
        for i, row in user_df.iterrows():
            node = row['POI_id']
            if node not in G.nodes():
                G.add_node(row['POI_id'],
                           checkin_cnt=1,
                           poi_catid=row['POI_catid'],
                           poi_catid_code=row['POI_catid_code'],
                           poi_catname=row['POI_catname'],
                           latitude=row['latitude'],
                           longitude=row['longitude'])
            else:
                G.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in seq)
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['POI_id']
            traj_id = row['trajectory_id']
            # No edge for the begin of the seq or different traj
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue

            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_traj_id = traj_id
            previous_poi_id = poi_id

    return G


def save_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.

    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    # np.save(os.path.join(dst_dir, 'adj_mtx.npy'), A.todense())
    np.savetxt(os.path.join(dst_dir, 'graph_A.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, 'graph_X.csv'), 'w') as f:
        print('node_name/poi_id,checkin_cnt,poi_catid,poi_catid_code,poi_catname,latitude,longitude', file=f)
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            poi_catid = each[1]['poi_catid']
            poi_catid_code = each[1]['poi_catid_code']
            poi_catname = each[1]['poi_catname']
            latitude = each[1]['latitude']
            longitude = each[1]['longitude']
            print(f'{node_name},{checkin_cnt},'
                  f'{poi_catid},{poi_catid_code},"{poi_catname}",'
                  f'{latitude},{longitude}', file=f)


def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, 'graph.pkl'), 'wb'))


def save_graph_edgelist(G, dst_dir):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join(dst_dir, 'graph_node_id2idx.txt'), 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node}, {i}', file=f)

    with open(os.path.join(dst_dir, 'graph_edge.edgelist'), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[src_node]} {node_id2idx[dst_node]} {weight}', file=f)


def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude', feature4='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X


def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    # Node degrees (mean and percentiles)
    node_degrees = [each[1] for each in G.degree]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    # Edge weights (mean and percentiles)
    edge_weights = []
    for n, nbrs in G.adj.items():
        for nbr, attr in nbrs.items():
            weight = attr['weight']
            edge_weights.append(weight)
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")

def geohash_encode(lat, lon, precision=6):
    """Encode latitude and longitude into geohash"""
    return geohash.encode(lat, lon, precision=precision)

def geohash_add2_df(df):
    """Add geohash to dataframe"""
    df['geohash'] = df.apply(lambda row: geohash_encode(row['latitude'], row['longitude']), axis=1)
    return df

def build_global_POI_geo_graph(df, exclude_user=None):
    G = nx.DiGraph()
    users = list(set(df['user_id'].to_list()))
    if exclude_user in users: users.remove(exclude_user)
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user_id'] == user_id]

        # Add node (POI)
        for i, row in user_df.iterrows():
            node = row['geohash']
            if node not in G.nodes():
                G.add_node(row['geohash'],
                           checkin_cnt=1,
                           )
            else:
                G.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in seq)
        previous_geo_hash = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            geo_hash = row['geohash']
            traj_id = row['trajectory_id']
            if (previous_geo_hash == 0) or (previous_traj_id != traj_id):
                previous_geo_hash = geo_hash
                previous_traj_id = traj_id
                continue

            # Add edges
            if G.has_edge(previous_geo_hash, geo_hash):
                G.edges[previous_geo_hash, geo_hash]['weight'] += 1
            else:
                G.add_edge(previous_geo_hash, geo_hash, weight=1)
            previous_traj_id = traj_id
            previous_geo_hash = geo_hash

    return G

def save_geo_graph_to_csv(G, dst_dir):
    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    np.savetxt(os.path.join(dst_dir, 'geo_graph_A.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())
    with open(os.path.join(dst_dir, 'geo_graph_X.csv'), 'w') as f:
        print('geohash\tcheckin_cnt', file=f)
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            print(f'{node_name}\t{checkin_cnt}', file=f)


if __name__ == '__main__':
    dst_dir = r'dataset_50/GB'

    # Build POI checkin trajectory graph
    train_df = pd.read_csv(os.path.join(dst_dir, 'GB_train.csv'))
    print('Build global POI checkin graph -----------------------------------')
    G = build_global_POI_checkin_graph(train_df)

    geohash_add2_df(train_df)
    G_geo = build_global_POI_geo_graph(train_df)

    # Save graph to disk
    save_graph_to_pickle(G, dst_dir=dst_dir)
    save_graph_to_csv(G, dst_dir=dst_dir)
    save_graph_edgelist(G, dst_dir=dst_dir)
    save_geo_graph_to_csv(G_geo, dst_dir=dst_dir)

