import numpy as np
import pandas as pd

from Geohash import geohash

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


def geohash_encode(lat, lon, precision=6):
    """Encode latitude and longitude into geohash."""
    return geohash.encode(lat, lon, precision=precision)

def load_graph_node_features_geo(path, feature1='checkin_cnt', feature2='cat_id',
                                feature3='latitude', feature4='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path) 
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()
    # Apply geohash encoding
    geohash = df.apply(lambda row: geohash_encode(row[feature3], row[feature4]), axis=1)

    # 为了保证每次结果一致，对unique()后的geohash进行排序
    geohash_unique_sorted = sorted(geohash.unique())

    # 创建一个唯一ID映射，由于现在是排序后的结果，将保持一致性
    geohash_to_id = {geohash: idx for idx, geohash in enumerate(geohash_unique_sorted)}

    # 映射geohash到ID
    geohash_id = geohash.map(geohash_to_id)
    geohash_list = geohash_id.tolist()

    return X,geohash_list

