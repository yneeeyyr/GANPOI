import logging
import logging
import os
import pathlib
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import load_graph_adj_mtx, load_graph_node_features, load_graph_node_features_geo
from model import GCN, NodeAttnMap, UserEmbeddings, CategoryEmbeddings, FuseEmbeddings, TransformerModel,Discriminator ,GeoHashEmbeddings, Time2Vec
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss, ndcg_last_timestep

from collections import OrderedDict

from sklearn.model_selection import GridSearchCV

import random
args=parameter_parser()
seed=args.seed

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)

    # Build POI graph (built from train_df)
    print('Loading POI graph...')
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    raw_X, geohash_list= load_graph_node_features_geo(args.data_node_feats,
                                     args.feature1,
                                     args.feature2,
                                     args.feature3,
                                     args.feature4)


    logging.info(
        f"raw_X.shape: {raw_X.shape}; " # (4980,1097)
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}.")
    logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")
    num_pois = raw_X.shape[0]
    num_geos = len(set(geohash_list))

    # One-hot encoding poi categories
    logging.info('One-hot encoding poi categories id')
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")
    logging.info(f'POI categories: {list(one_hot_encoder.categories_[0])}')

    # Save ont-hot encoder
    with open(os.path.join(args.save_dir, 'one-hot-encoder.pkl'), "wb") as f:
        pickle.dump(one_hot_encoder, f)

    # Normalization
    print('Laplician matrix...')
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')

    # POI id to index
    nodes_df = pd.read_csv(args.data_node_feats)
    poi_ids = sorted(list(set(nodes_df['node_name/poi_id'].tolist())))
    poi_id2idx_dict = OrderedDict(zip(poi_ids, range(len(poi_ids))))

    # Cat id to index
    cat_ids = sorted(list(set(nodes_df[args.feature2].tolist())))
    cat_id2idx_dict = OrderedDict(zip(cat_ids, range(len(cat_ids))))

    # Poi idx to cat idx
    poi_idx2cat_idx_dict = {}
    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = \
            cat_id2idx_dict[row[args.feature2]]

    # Poi idx to geo idx
    geo_ids = sorted(list(set(geohash_list)))
    geo_id2idx_dict=OrderedDict(zip(geo_ids,range(len(geo_ids))))
    poi_idx2geohash_id = {idx: geohash_list[idx] for idx in range(len(geohash_list))}

    # User id to index
    user_ids = sorted(list(set(train_df['user_id'].tolist())))
    user_ids = list(map(str, user_ids))

    user_id2idx_dict = OrderedDict(zip(user_ids, range(len(user_ids))))

    # Print user-trajectories count
    traj_list = list(set(train_df['trajectory_id'].tolist()))

    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []  # traj id: user id + traj no.
            self.input_seqs = []
            self.label_seqs = []

            #for traj_id in tqdm(set(train_df['trajectory_id'].tolist())):
            traj_ids = sorted(train_df['trajectory_id'].unique().tolist())
            for traj_id in tqdm(traj_ids):
                traj_df = train_df[train_df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
                time_feature = traj_df[args.time_feature].to_list()

                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

                if len(input_seq) < args.short_traj_thres:
                    continue

                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []
            
            traj_ids = sorted(df['trajectory_id'].unique().tolist())

            #for traj_id in tqdm(set(df['trajectory_id'].tolist())):
            for traj_id in tqdm(traj_ids):
                user_id = traj_id.split('_')[0]

                # Ignore user if not in training set
                if user_id not in user_id2idx_dict.keys():
                    continue

                # Ger POIs idx in this trajectory
                traj_df = df[df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = []
                time_feature = traj_df[args.time_feature].to_list()

                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        # Ignore poi if not in training set
                        continue

                # Construct input seq and label seq
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)

    # %% ====================== Build Models ======================
    # Model1: POI embedding model
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)

    args.gcn_nfeat = X.shape[1]
    poi_embed_model = GCN(ninput=args.gcn_nfeat,
                          nhid=args.gcn_nhid,
                          noutput=args.poi_embed_dim,
                          dropout=args.gcn_dropout)

    # Node Attn Model
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)

    # %% Model2: User embedding model, nn.embedding
    num_users = len(user_id2idx_dict)
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)

    # %% Model3: Time Model
    time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)

    # %% Model4: Category embedding model
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)

    # %% Model5: Embedding fusion models
    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
    # embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)
    embed_fuse_model2 = FuseEmbeddings(args.geo_embed_dim, args.cat_embed_dim)
    # embed_fuse_model2 = FuseEmbeddings(args.geo_embed_dim, args.time_embed_dim)

    # %% Model6: geo fusion models
    geo_embed_model=GeoHashEmbeddings(num_geos, args.geo_embed_dim)
    
    # %% Model7: gan model
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.cat_embed_dim+ args.time_embed_dim + args.geo_embed_dim 

    seq_model = TransformerModel(num_pois,
                                num_cats,
                                args.seq_input_embed,
                                args.transformer_nhead,
                                args.transformer_nhid,
                                args.transformer_nlayers,
                                dropout=args.transformer_dropout)

    dis_model = Discriminator(num_pois,
                              num_cats,
                              num_geos,
                              args.seq_input_embed,
                              args.transformer_nhead,
                              args.transformer_nhid,
                              args.transformer_nlayers,
                              dropout=args.transformer_dropout)
    
    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                    list(node_attn_model.parameters()) +
                                    list(user_embed_model.parameters()) +
                                    list(time_embed_model.parameters()) +
                                    list(cat_embed_model.parameters()) +
                                    list(geo_embed_model.parameters()) + 
                                    list(seq_model.parameters()) +
                                    list(dis_model.parameters()),                                                 
                            lr=args.lr,
                            weight_decay=args.weight_decay)

    criterion_bce = nn.CrossEntropyLoss(ignore_index=-1)#nn.CrossEntropyLoss(ignore_index=-1) nn.BCELoss()
    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_geo = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_time = maksed_mse_loss

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # %% Tool functions for training
    def input_traj_to_embeddings(sample, poi_embeddings):
        # Parse sample
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        label_seq = [each[0] for each in sample[2]] #
        input_seq_time = [each[1] for each in sample[1]]
        label_seq_time = [each[1] for each in sample[2]]
        input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]
        label_seq_cat = [poi_idx2cat_idx_dict[each] for each in label_seq]

        input_seq_geohash = [poi_idx2geohash_id[each] for each in input_seq]
        label_seq_geohash = [poi_idx2geohash_id[each] for each in label_seq]

        # User to embedding
        user_id = traj_id.split('_')[0]
        user_idx = user_id2idx_dict[user_id]
        input = torch.LongTensor([user_idx]).to(device=args.device)
        user_embedding = user_embed_model(input)
        user_embedding = torch.squeeze(user_embedding)

        # POI to embedding and fuse embeddings
        input_seq_embed = []
        label_seq_embed = []

        for idx in range(len(input_seq)):
            poi_embedding = poi_embeddings[input_seq[idx]]
            poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)

            # Time to vector
            time_embedding = time_embed_model(
                torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
            time_embedding = torch.squeeze(time_embedding).to(device=args.device)

            # Categroy to embedding
            cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=args.device)
            cat_embedding = cat_embed_model(cat_idx)
            cat_embedding = torch.squeeze(cat_embedding)
        
            #geohash to embedding
            geohash_idx = torch.LongTensor([input_seq_geohash[idx]]).to(device=args.device)
            geohash_embedding = geo_embed_model(geohash_idx)
            geohash_embedding = torch.squeeze(geohash_embedding)

            # Fuse user+poi embeds
            fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)
            # fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)

            fused_embedding2 = embed_fuse_model2(geohash_embedding, cat_embedding)
            # fused_embedding2 = embed_fuse_model2(geohash_embedding, time_embedding)

            # Concat time, cat after user+poi
            # concat_embedding = torch.cat((fused_embedding1, fused_embedding2),dim=-1)
            # concat_embedding = torch.cat((user_embedding, poi_embedding), dim=-1)
            concat_embedding=torch.cat((fused_embedding1, fused_embedding2, time_embedding), dim=-1)

            # Save final embed
            input_seq_embed.append(concat_embedding)

        for idx in range(len(label_seq)):
            poi_embedding = poi_embeddings[label_seq[idx]]
            poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)

            # Time to vector
            time_embedding = time_embed_model(
                torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
            time_embedding = torch.squeeze(time_embedding).to(device=args.device)
        
            # # Categroy to embedding
            cat_idx = torch.LongTensor([label_seq_cat[idx]]).to(device=args.device)
            cat_embedding = cat_embed_model(cat_idx)
            cat_embedding = torch.squeeze(cat_embedding)

            # Geohash to embedding
            geohash_idx = torch.LongTensor([input_seq_geohash[idx]]).to(device=args.device)
            geohash_embedding = geo_embed_model(geohash_idx)
            geohash_embedding = torch.squeeze(geohash_embedding)

            # Fuse user+poi embeds
            fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)
            # fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)
            # fused_embedding2 = embed_fuse_model2(geohash_embedding, time_embedding)
            fused_embedding2 = embed_fuse_model2(geohash_embedding, cat_embedding)

            # Concat time, cat after user+poi
            # concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)
            # concat_embedding = torch.cat((user_embedding, poi_embedding), dim=-1)
            
            concat_embedding=torch.cat((fused_embedding1, fused_embedding2, time_embedding), dim=-1)

            # Save final embed
            label_seq_embed.append(concat_embedding)

        return input_seq_embed, label_seq_embed

    def adjust_pred_prob_by_graph(y_pred_poi):
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
        attn_map = node_attn_model(X, A)

        for i in range(len(batch_seq_lens)):
            traj_i_input = batch_input_seqs[i]  # list of input check-in pois
            for j in range(len(traj_i_input)):
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

        return y_pred_poi_adjusted

    # %% ====================== Train Loop ======================
    poi_embed_model = poi_embed_model.to(device=args.device)
    geo_embed_model = geo_embed_model.to(device=args.device)
    node_attn_model = node_attn_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    seq_model = seq_model.to(device=args.device)
    dis_model = dis_model.to(device=args.device)

    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_ndcg1_list = [] 
    train_epochs_ndcg5_list = [] 
    train_epochs_ndcg10_list = [] 
    train_epochs_ndcg20_list = [] 
    train_epochs_mrr_list = []
    train_epochs_d_loss_list = []
    train_epochs_g_loss_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_time_loss_list = []
    train_epochs_cat_loss_list = []
    train_epochs_geo_loss_list = []
    
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_ndcg1_list = []
    val_epochs_ndcg5_list = []
    val_epochs_ndcg10_list = []
    val_epochs_ndcg20_list = []
    val_epochs_mrr_list = []
    val_epochs_d_loss_list = []
    val_epochs_g_loss_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_time_loss_list = []
    val_epochs_cat_loss_list = []
    val_epochs_geo_loss_list = []
    
    # For saving ckpt
    max_val_score = -np.inf

    # For early stopping
    best_val_score = 0
    best_val_epoch = 0
    best_val_top1_acc = 0
    best_val_top5_acc = 0
    best_val_top10_acc = 0
    best_val_top20_acc = 0
    best_val_mAP20 = 0
    best_val_ndcg1 = 0
    best_val_ndcg5 = 0
    best_val_ndcg10 = 0
    best_val_ndcg20 = 0
    best_val_mrr = 0
    previous_val_top20_acc = 0
    previous_val_ndcg20 = 0
    patience_times = 0

    last_val_score = 0
    last_val_epoch = 0
    last_val_top1_acc = 0
    last_val_top5_acc = 0
    last_val_top10_acc = 0
    last_val_top20_acc = 0
    last_val_mAP20 = 0
    last_val_ndcg1 = 0
    last_val_ndcg5 = 0
    last_val_ndcg10 = 0
    last_val_ndcg20 = 0
    last_val_mrr = 0

# %% ====================== Train ======================
    #gan_model.train()
    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_embed_model.train()

        geo_embed_model.train()

        node_attn_model.train()
        user_embed_model.train()
        time_embed_model.train()
        cat_embed_model.train()
        seq_model.train()
        dis_model.train()

        train_batches_g_loss_list = []
        train_batches_d_loss_list = []
        train_batches_loss_list = []

        train_batches_poi_loss_list = []
        train_batches_time_loss_list = []
        train_batches_cat_loss_list = []
        train_batches_geo_loss_list = []  

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_ndcg1_list = []
        train_batches_ndcg5_list = []
        train_batches_ndcg10_list = []
        train_batches_ndcg20_list = []
        train_batches_mrr_list = []
        
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        for b_idx, batch in enumerate(train_loader):
            if len(batch) != args.batch: #如果当前批次的大小与设定的批次大小不一致，则重新生成适应当前批次大小的掩码
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)
            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_pos_embed = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []
            batch_seq_labels_geo = []
            batch_seq_targets=[]

            poi_embeddings = poi_embed_model(X, A)

            # Convert input seq to embeddings
            for sample in batch:
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                label_seq_geos = [poi_idx2geohash_id[each] for each in label_seq]

                input_seq_embed, label_seq_embed=input_traj_to_embeddings(sample, poi_embeddings)
                input_seq_embed = torch.stack(input_seq_embed)
                label_seq_embed = torch.stack(label_seq_embed)

                batch_seq_embeds.append(input_seq_embed) #将当前样本的输入嵌入序列添加到batch_seq_embeds列表中，存储当前批次中所有样本的嵌入表示
                batch_pos_embed.append(label_seq_embed) #n
                
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
                batch_seq_labels_geo.append(torch.LongTensor(label_seq_geos))
                
            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            batch_padded_pos = pad_sequence(batch_pos_embed, batch_first=True, padding_value=-1) #n
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
            label_padded_geo = pad_sequence(batch_seq_labels_geo, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            x1=batch_padded_pos.to(device=args.device, dtype=torch.float) #n
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_geo = label_padded_geo.to(device=args.device, dtype=torch.long)
            
            #optimizer_d.zero_grad()

            fake_features = seq_model(x, src_mask)    
            # g_loss=criterion_bce(fake_features.transpose(1, 2), x1.mean(dim=-1).long())
            recon_loss=criterion_bce(fake_features.transpose(1, 2), x1.mean(dim=-1).long())
            # 判别器的输出
            fake_poi, fake_geo, fake_cat, fake_time = dis_model(fake_features, src_mask)#
            # 使用图注意力机制调整预测结果
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(fake_poi)           
            # Fake data loss
            fake_loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            fake_loss_time = criterion_time(torch.squeeze(fake_time), y_time)#torch.zeros_like(fake_time))
            fake_loss_cat = criterion_cat(fake_cat.transpose(1, 2), y_cat)
            fake_loss_geo = criterion_geo(fake_geo.transpose(1, 2), y_geo)
            # d_loss_fake = fake_loss_poi + fake_loss_cat + fake_loss_geo * args.geo_loss_weight + fake_loss_time * args.time_loss_weight 
            adv_loss = fake_loss_poi + fake_loss_time * args.time_loss_weight+ fake_loss_cat + fake_loss_geo * args.geo_loss_weight
            g_loss=recon_loss+adv_loss*args.adv_loss_weight
            #对真实数据进行分类
            poi_pred_real, geo_pred_real, cat_pred_real, time_pred_real = dis_model(x, src_mask) #
            y_pred_poi_adjuste_real = adjust_pred_prob_by_graph(poi_pred_real)    
            real_loss_poi = criterion_poi(poi_pred_real.transpose(1, 2), y_poi)
            real_loss_time = criterion_time(torch.squeeze(time_pred_real), y_time)
            real_loss_cat = criterion_cat(cat_pred_real.transpose(1, 2), y_cat)
            real_loss_geo = criterion_geo(geo_pred_real.transpose(1, 2), y_geo)
            d_loss_real = real_loss_poi + real_loss_time * args.time_loss_weight + real_loss_cat+ real_loss_geo * args.geo_loss_weight 
            loss_time=fake_loss_time+real_loss_time
            loss_cat=fake_loss_cat+fake_loss_cat
            loss_poi=fake_loss_poi+real_loss_poi
            loss_geo=fake_loss_geo+real_loss_geo

            d_loss_fake=adv_loss
            d_loss = d_loss_fake + d_loss_real
            
            joint_loss = g_loss * args.gen_loss_weight + d_loss * args.dis_loss_weight

            optimizer.zero_grad()                             
            joint_loss.backward()
            optimizer.step()
                     
            # Performance measurement 初始化指标的累加器
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            ndcg1 = 0
            ndcg5 = 0
            ndcg10 = 0
            ndcg20 = 0
            mrr = 0

            #将标签和预测从GPU移动到CPU并转换为NumPy数组
            batch_label_pois = y_poi.detach().cpu().numpy() #当前批次的实际poi标签。.detach()从计算图中分离该张量，使其不再与计算图关联，从而避免后续的梯度计算
            batch_pred_pois = y_pred_poi_adjuste_real.detach().cpu().numpy() #存储当前批次调整后的 POI 预测，
            batch_pred_times = time_pred_real.detach().cpu().numpy() #存储当前批次对时间的预测结果
            batch_pred_cats = cat_pred_real.detach().cpu().numpy() #存储当前批次对类别的预测结果
            batch_pred_geos = geo_pred_real.detach().cpu().numpy() #存储当前批次对类别的预测结果

            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens): #使用zip函数并行遍历实际标签，预测结果和序列长度
                label_pois = label_pois[:seq_len]  # shape: (seq_len, ) 根据当前样本的有效序列长度截取实际标签
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi) 根据有效序列长度截取预测结果
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1) #评估模型在当前样本的top-1预测准确性 （utils 133行）
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20) # 计算当前样本的平均精度（utils 145行）
                ndcg1 += ndcg_last_timestep(label_pois, pred_pois, k=1)
                ndcg5 += ndcg_last_timestep(label_pois, pred_pois, k=5)
                ndcg10 += ndcg_last_timestep(label_pois, pred_pois, k=10)
                ndcg20 += ndcg_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois) #计算当前样本的平均倒数排名 (utils 160)
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois)) #存储当前批次的top-1准确率
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois)) #存储当前批次的mAP20
            train_batches_ndcg1_list.append(ndcg1 / len(batch_label_pois))
            train_batches_ndcg5_list.append(ndcg5 / len(batch_label_pois))
            train_batches_ndcg10_list.append(ndcg10 / len(batch_label_pois))
            train_batches_ndcg20_list.append(ndcg20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois)) #存储当前批次的MRR
            train_batches_d_loss_list.append(d_loss.detach().cpu().numpy()) #存储当前批次的损失值
            train_batches_g_loss_list.append(g_loss.detach().cpu().numpy()) #存储当前批次的损失值
            train_batches_loss_list.append(joint_loss.detach().cpu().numpy()) #存储当前批次的损失值
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            train_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())
            train_batches_geo_loss_list.append(loss_geo.detach().cpu().numpy())

            # Log the training progress
            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0 #用于选择当前批次的第一个样本以便于记录其详细信息
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, Loss D: {d_loss.item()}, Loss G: {g_loss.item()},Loss : {joint_loss.item()},'#
                             f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, ' #获取当前批次的平均top-1准确率，并格式化为两位小数
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                             f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                             f'train_move_time_loss:{np.mean(train_batches_time_loss_list):.2f}\n'
                             f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n' 
                             f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                             f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                             f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                             f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                             f'train_move_ndcg1:{np.mean(train_batches_ndcg1_list):.4f}\n'
                             f'train_move_ndcg5:{np.mean(train_batches_ndcg5_list):.4f}\n'
                             f'train_move_ndcg10:{np.mean(train_batches_ndcg10_list):.4f}\n'
                             f'train_move_ndcg20:{np.mean(train_batches_ndcg20_list):.4f}\n'
                             f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' #+记录经过注意力机制调整后的poi预测序列
                             f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n' #将当前样本的标签转换为类别索引并记录
                             f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' #记录模型对类别的预测序列
                             f'label_seq_geo:{[poi_idx2geohash_id[each[0]] for each in batch[sample_idx][2]]}\n' #将当前样本的标签转换为类别索引并记录
                             f'pred_seq_geo:{list(np.argmax(batch_pred_geos, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' #+ 记录模型对类别的预测序列
                             f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n' #将当前样本的时间标签记录下来
                             f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100) #记录模型对时间的预测结果，并输出分隔符以便于日志的可读性

        # train end --------------------------------------------------------------------------------------------------------
        #将模型设置为评估模式
        poi_embed_model.eval()
        node_attn_model.eval()
        user_embed_model.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        geo_embed_model.eval()
        embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        seq_model.eval()
        dis_model.eval()

        #创建空列表，用于存储验证过程中的各类评估指标
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_ndcg1_list = []
        val_batches_ndcg5_list = []
        val_batches_ndcg10_list = []
        val_batches_ndcg20_list = []
        val_batches_mrr_list = []
        
        val_batches_poi_loss_list = []
        val_batches_time_loss_list = []
        val_batches_cat_loss_list = []
        val_batches_geo_loss_list = []

        val_batches_g_loss_list = []
        val_batches_d_loss_list = []
        val_batches_loss_list = []

        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device) #生成用于序列模型的掩码
        for vb_idx, batch in enumerate(val_loader):
            if len(batch) != args.batch: #如果当前批次的大小与设定的批次大小不一致，则重新生成适应当前批次大小的掩码
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)
            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_pos_embed = [] #
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []
            batch_seq_labels_geo = []
            
            poi_embeddings = poi_embed_model(X, A)

            # Convert input seq to embeddings
            for sample in batch:
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                label_seq_geos = [poi_idx2geohash_id[each] for each in label_seq]

                input_seq_embed, label_seq_embed = input_traj_to_embeddings(sample, poi_embeddings)
                input_seq_embed = torch.stack(input_seq_embed)
                label_seq_embed = torch.stack(label_seq_embed)

                #input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings))
                batch_seq_embeds.append(input_seq_embed) ##将当前样本的输入嵌入序列添加到batch_seq_embeds列表中，存储当前批次中所有样本的嵌入表示
                batch_pos_embed.append(label_seq_embed)

                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
                batch_seq_labels_geo.append(torch.LongTensor(label_seq_geos))

            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            batch_padded_pos = pad_sequence(batch_pos_embed, batch_first=True, padding_value=-1) #n
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
            label_padded_geo = pad_sequence(batch_seq_labels_geo, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            x1=batch_padded_pos.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_geo = label_padded_geo.to(device=args.device, dtype=torch.long)

            # Generate fake features using generator
            fake_features= seq_model(x, src_mask)
            recon_loss=criterion_bce(fake_features.transpose(1, 2), x1.mean(dim=-1).long())
            
            fake_poi, fake_geo, fake_cat , fake_time= dis_model(fake_features,src_mask) # 
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(fake_poi)
            # Fake data loss
            fake_loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            fake_loss_time = criterion_time(torch.squeeze(fake_time), y_time)
            fake_loss_cat = criterion_cat(fake_cat.transpose(1, 2), y_cat)
            fake_loss_geo = criterion_geo(fake_geo.transpose(1, 2), y_geo)
            #d_loss_fake = fake_loss_poi + fake_loss_cat + fake_loss_geo * args.geo_loss_weight + fake_loss_time * args.time_loss_weight 
            adv_loss=fake_loss_poi  + fake_loss_time * args.time_loss_weight+ fake_loss_cat + fake_loss_geo * args.geo_loss_weight
            g_loss=recon_loss+adv_loss*args.adv_loss_weight
            #对真实数据进行分类
            poi_pred_real, geo_pred_real, cat_pred_real, time_pred_real  = dis_model(x, src_mask)#
            y_pred_poi_adjusted_real = adjust_pred_prob_by_graph(poi_pred_real)
            real_loss_poi = criterion_poi(poi_pred_real.transpose(1, 2), y_poi)
            real_loss_time = criterion_time(torch.squeeze(time_pred_real), y_time) #torch.zeros_like(fake_time))
            real_loss_cat = criterion_cat(cat_pred_real.transpose(1, 2), y_cat)
            real_loss_geo = criterion_geo(geo_pred_real.transpose(1, 2), y_geo)
            d_loss_real = real_loss_poi + real_loss_cat  + real_loss_time * args.time_loss_weight+ real_loss_geo * args.geo_loss_weight 
            loss_time=fake_loss_time+real_loss_time
            loss_cat=fake_loss_cat+fake_loss_cat
            loss_poi=fake_loss_poi+real_loss_poi
            loss_geo=fake_loss_geo+real_loss_geo

            d_loss_fake=adv_loss
            d_loss = d_loss_fake + d_loss_real
        
            joint_loss = g_loss * args.gen_loss_weight+ d_loss * args.dis_loss_weight# 
        
            # Performance measurement 初始化指标
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            ndcg1 = 0
            ndcg5 = 0
            ndcg10 = 0
            ndcg20 = 0
            mrr = 0
            #将标签和预测从GPU移动到CPU并转换为NumPy数组
            batch_label_pois = y_poi.detach().cpu().numpy() #当前批次的实际poi标签。.detach()从计算图中分离该张量，使其不再与计算图关联，从而避免后续的梯度计算
            batch_pred_pois = y_pred_poi_adjusted_real.detach().cpu().numpy() #存储当前批次调整后的 POI 预测，
            batch_pred_times = time_pred_real.detach().cpu().numpy() #存储当前批次对时间的预测结果
            batch_pred_cats = cat_pred_real.detach().cpu().numpy() #存储当前批次对类别的预测结果
            batch_pred_geos = geo_pred_real.detach().cpu().numpy() #存储当前批次对类别的预测结果

            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens): #使用zip函数并行遍历实际标签，预测结果和序列长度
                label_pois = label_pois[:seq_len]  # shape: (seq_len, ) 根据当前样本的有效序列长度截取实际标签
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi) 根据有效序列长度截取预测结果
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1) #评估模型在当前样本的top-1预测准确性 （utils 133行）
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20) # 计算当前样本的平均精度 （utils 145行）
                ndcg1 += ndcg_last_timestep(label_pois, pred_pois, k=1)
                ndcg5 += ndcg_last_timestep(label_pois, pred_pois, k=5)
                ndcg10 += ndcg_last_timestep(label_pois, pred_pois, k=10)
                ndcg20 += ndcg_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois) #计算当前样本的平均倒数排名 (utils 160)
            #计算当前批次的平均值表并存储
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_ndcg1_list.append(ndcg1 / len(batch_label_pois))
            val_batches_ndcg5_list.append(ndcg5 / len(batch_label_pois))
            val_batches_ndcg10_list.append(ndcg10 / len(batch_label_pois))
            val_batches_ndcg20_list.append(ndcg20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_d_loss_list.append(d_loss.detach().cpu().numpy()) 
            val_batches_g_loss_list.append(g_loss.detach().cpu().numpy()) 
            val_batches_loss_list.append(joint_loss.detach().cpu().numpy()) 
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            val_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            val_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())
            val_batches_geo_loss_list.append(loss_geo.detach().cpu().numpy())

            # Log the training progress
            if (vb_idx % (args.batch * 2)) == 0:
                sample_idx = 0 #用于选择当前批次的第一个样本以便于记录其详细信息
                logging.info(f'Epoch:{epoch}, batch:{vb_idx}, Loss D: {d_loss.item()}, Loss G: {g_loss.item()},Loss: {joint_loss.item()},'#
                             f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, ' #获取当前批次的平均top-1准确率，并格式化为两位小数
                             f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                             f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                             f'val_move_time_loss:{np.mean(val_batches_time_loss_list):.2f} \n'
                             f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n' #top-1准确率的平均值
                             f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                             f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                             f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                             f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n' #平均精度的平均值
                             f'val_move_ndcg1:{np.mean(val_batches_ndcg1_list):.4f} \n'
                             f'val_move_ndcg5:{np.mean(val_batches_ndcg5_list):.4f} \n'
                             f'val_move_ndcg10:{np.mean(val_batches_ndcg10_list):.4f} \n'
                             f'val_move_ndcg20:{np.mean(val_batches_ndcg20_list):.4f} \n'
                             f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n' #平均倒数排名的平均值
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'# +记录经过注意力调整的POI预测结果
                             f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n' #当前样本的类别标签序列
                             f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' #当前样本的时间标签序列
                             f'label_seq_geo:{[poi_idx2geohash_id[each[0]] for each in batch[sample_idx][2]]}\n' #当前样本的类别标签序列
                             f'pred_seq_geo:{list(np.argmax(batch_pred_geos, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' # +当前样本的时间标签序列
                             f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'  #记录当前样本的时间标签序列
                             f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)#模型的时间预测结果
        # valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics 所有批次的平均值
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list) 
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_ndcg1 = np.mean(train_batches_ndcg1_list)
        epoch_train_ndcg5 = np.mean(train_batches_ndcg5_list)
        epoch_train_ndcg10 = np.mean(train_batches_ndcg10_list)
        epoch_train_ndcg20 = np.mean(train_batches_ndcg20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_d_loss = np.mean(train_batches_d_loss_list)
        epoch_train_g_loss = np.mean(train_batches_g_loss_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_train_time_loss = np.mean(train_batches_time_loss_list)
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)
        epoch_train_geo_loss = np.mean(train_batches_geo_loss_list)

        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_ndcg1 = np.mean(val_batches_ndcg1_list)
        epoch_val_ndcg5 = np.mean(val_batches_ndcg5_list)
        epoch_val_ndcg10 = np.mean(val_batches_ndcg10_list)
        epoch_val_ndcg20 = np.mean(val_batches_ndcg20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_d_loss = np.mean(val_batches_d_loss_list)
        epoch_val_g_loss = np.mean(val_batches_g_loss_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        epoch_val_time_loss = np.mean(val_batches_time_loss_list)
        epoch_val_cat_loss = np.mean(val_batches_cat_loss_list)
        epoch_val_geo_loss = np.mean(val_batches_geo_loss_list)
        
        # Save metrics to list 将每个训练周期计算得到的平均性能指标和损失值保存到对应的列表中
        train_epochs_d_loss_list.append(epoch_train_d_loss)
        train_epochs_g_loss_list.append(epoch_train_g_loss)
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_time_loss_list.append(epoch_train_time_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_geo_loss_list.append(epoch_train_geo_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_ndcg1_list.append(epoch_train_ndcg1)
        train_epochs_ndcg5_list.append(epoch_train_ndcg5)
        train_epochs_ndcg10_list.append(epoch_train_ndcg10)
        train_epochs_ndcg20_list.append(epoch_train_ndcg20)
        train_epochs_mrr_list.append(epoch_train_mrr)

        val_epochs_d_loss_list.append(epoch_val_d_loss)
        val_epochs_g_loss_list.append(epoch_val_g_loss)
        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        val_epochs_time_loss_list.append(epoch_val_time_loss)
        val_epochs_cat_loss_list.append(epoch_val_cat_loss)
        val_epochs_geo_loss_list.append(epoch_val_geo_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_ndcg1_list.append(epoch_val_ndcg1)
        val_epochs_ndcg5_list.append(epoch_val_ndcg5)
        val_epochs_ndcg10_list.append(epoch_val_ndcg10)
        val_epochs_ndcg20_list.append(epoch_val_ndcg20)
        val_epochs_mrr_list.append(epoch_val_mrr)

        # Monitor loss and score
        monitor_g_loss = epoch_val_g_loss
        monitor_d_loss = epoch_val_d_loss #将验证集上的平均损失值赋值给monitor_loss，用于调整学习率
        monitor_loss = epoch_val_loss
        monitor_score = np.mean(epoch_val_top1_acc * 4 + epoch_val_top20_acc) #计算一个综合评分，用于衡量模型在验证集上的整体表现

        # early stopping
        if epoch_val_top20_acc > best_val_top20_acc:
            best_val_loss = epoch_val_loss
            best_val_top1_acc = epoch_val_top1_acc
            best_val_top5_acc = epoch_val_top5_acc
            best_val_top10_acc = epoch_val_top10_acc
            best_val_top20_acc = epoch_val_top20_acc
            best_val_mAP20 = epoch_val_mAP20
            best_val_ndcg1 = epoch_val_ndcg1
            best_val_ndcg5 = epoch_val_ndcg5
            best_val_ndcg10 = epoch_val_ndcg10
            best_val_ndcg20 = epoch_val_ndcg20
            best_val_mrr = epoch_val_mrr
            best_val_epoch = epoch

        now_score = epoch_val_top20_acc + epoch_val_ndcg20
        previous_score = previous_val_top20_acc + previous_val_ndcg20
        if now_score == previous_score:
            patience_times += 1
        else:
            patience_times = 0
        if patience_times >= 5:
            logging.info(f"Early stopping at epoch {epoch}")
            break

        previous_val_top20_acc = epoch_val_top20_acc
        previous_val_ndcg20 = epoch_val_ndcg20

        # Adjust learning rate based on loss
        lr_scheduler.step(monitor_loss)
        #gen_lr_scheduler.step(monitor_g_loss)
        #dis_lr_scheduler.step(monitor_d_loss)

        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n" #记录当前的训练周期以及总周期数
                     f"train_d_loss:{epoch_train_d_loss:.4f}, "
                     f"train_g_loss:{epoch_train_g_loss:.4f}, " #训练集上的平均总损失
                     f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
                     f"train_time_loss:{epoch_train_time_loss:.4f}, "
                     f"train_cat_loss:{epoch_train_cat_loss:.4f}, "
                     f"train_geo_loss:{epoch_train_geo_loss:.4f}, "
                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mAP20:{epoch_train_mAP20:.4f}, "
                     f"train_ndcg1:{epoch_train_ndcg1:.4f}, "
                     f"train_ndcg5:{epoch_train_ndcg5:.4f}, "
                     f"train_ndcg10:{epoch_train_ndcg10:.4f}, "
                     f"train_ndcg20:{epoch_train_ndcg20:.4f}, "
                     f"train_mrr:{epoch_train_mrr:.4f}\n"
                     f"val_d_loss: {epoch_val_d_loss:.4f}, "
                     f"val_g_loss: {epoch_val_g_loss:.4f}, " #验证集上的平均总损失
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"val_poi_loss: {epoch_val_poi_loss:.4f}, "
                     f"val_time_loss: {epoch_val_time_loss:.4f}, "
                     f"val_cat_loss: {epoch_val_cat_loss:.4f}, "
                     f"val_geo_loss: {epoch_val_geo_loss:.4f}, "
                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mAP20:{epoch_val_mAP20:.4f}, "
                     f"val_ndcg1:{epoch_val_ndcg1:.4f}, "
                     f"val_ndcg5:{epoch_val_ndcg5:.4f}, "
                     f"val_ndcg10:{epoch_val_ndcg10:.4f}, "
                     f"val_ndcg20:{epoch_val_ndcg20:.4f}, "
                     f"val_mrr:{epoch_val_mrr:.4f}")

        if args.save_embeds:
            embeddings_save_dir = os.path.join(args.save_dir, 'embeddings') #指定嵌入向量的保存目录。如果目录不存在，则创建该目录
            if not os.path.exists(embeddings_save_dir): os.makedirs(embeddings_save_dir)
            # Save best epoch embeddings
            if monitor_score >= max_val_score: #如果当前周期的评分大于或等于历史最高评分max_val_score，则保存嵌入向量
                # Save poi embeddings
                poi_embeddings = poi_embed_model(X, A).detach().cpu().numpy() #使用poi_embed_model声测韩国的poi嵌入向量转换为numpy数组
                poi_embedding_list = [] #初始化一个空列表，用于存储每个poi的嵌入向量
                for poi_idx in range(len(poi_id2idx_dict)): #循环遍历所有poi
                    poi_embedding = poi_embeddings[poi_idx] #获取当前poi的嵌入向量
                    poi_embedding_list.append(poi_embedding) #将当前poi的嵌入向量添加到列表中
                save_poi_embeddings = np.array(poi_embedding_list) #将列表转换为nunpy数组
                np.save(os.path.join(embeddings_save_dir, 'saved_poi_embeddings'), save_poi_embeddings) #将poi嵌入向量保存为npy文件
                # Save user embeddings
                user_embedding_list = [] #初始化一个空列表，用于存储每个用户的嵌入向量
                for user_idx in range(len(user_id2idx_dict)): #循环遍历所有用户
                    input = torch.LongTensor([user_idx]).to(device=args.device) #将用户索引转换为张量并移动到指定设备
                    user_embedding = user_embed_model(input).detach().cpu().numpy().flatten() #生成用户嵌入向量，并转换为numpy数组
                    user_embedding_list.append(user_embedding) #将用户嵌入向量添加到列表中
                user_embeddings = np.array(user_embedding_list) #将列表转换为numpy数组
                np.save(os.path.join(embeddings_save_dir, 'saved_user_embeddings'), user_embeddings) #将用户嵌入向量保存为npy文件
                # Save cat embeddings
                cat_embedding_list = [] #初始化一个空列表，存储每个类别的嵌入向量
                for cat_idx in range(len(cat_id2idx_dict)): #循环遍历所有类别
                    input = torch.LongTensor([cat_idx]).to(device=args.device) #将类别索引转换为张量
                    cat_embedding = cat_embed_model(input).detach().cpu().numpy().flatten() #生成类别嵌入向量并将其转换为numpy数组
                    cat_embedding_list.append(cat_embedding)#将类别嵌入向量添加到列表中
                cat_embeddings = np.array(cat_embedding_list) #将列表转换为numpy数组
                np.save(os.path.join(embeddings_save_dir, 'saved_cat_embeddings'), cat_embeddings) #将类别嵌入向量保存为npy文件
                # Save geo embeddings
                geo_embedding_list = [] #初始化一个空列表，存储每个类别的嵌入向量
                for geo_idx in range(num_geos): #循环遍历所有类别
                    input = torch.LongTensor([geo_idx]).to(device=args.device) #将类别索引转换为张量
                    geo_embedding = geo_embed_model(input).detach().cpu().numpy().flatten() #生成类别嵌入向量并将其转换为numpy数组
                    geo_embedding_list.append(geo_embedding)#将类别嵌入向量添加到列表中
                geo_embeddings = np.array(geo_embedding_list) #将列表转换为numpy数组
                np.save(os.path.join(embeddings_save_dir, 'saved_geo_embeddings'), geo_embeddings) #将类别嵌入向量保存为npy文件
                # Save time embeddings
                time_embedding_list = [] #初始化一个空列表，用于存储每个事件单位的嵌入向量
                for time_idx in range(args.time_units): #time_units表示总的时间单位数
                    input = torch.FloatTensor([time_idx]).to(device=args.device) #价格时间单位转换为float型
                    time_embedding = time_embed_model(input).detach().cpu().numpy().flatten() #生成时间单位的嵌入向量
                    time_embedding_list.append(time_embedding) ##将时间嵌入向量添加到列表中
                time_embeddings = np.array(time_embedding_list)#将列表转换为numpy数组
                np.save(os.path.join(embeddings_save_dir, 'saved_time_embeddings'), time_embeddings)#将时间嵌入向量保存为npy文件

        # Save model state dict 保存模型的状态字典（state dict）以及相关的训练和验证指标
        if args.save_weights:
            state_dict = {
                'epoch': epoch, #当前训练周期
                'poi_embed_state_dict': poi_embed_model.state_dict(), 
                'node_attn_state_dict': node_attn_model.state_dict(),
                'user_embed_state_dict': user_embed_model.state_dict(),
                'time_embed_state_dict': time_embed_model.state_dict(),
                'cat_embed_state_dict': cat_embed_model.state_dict(),
                'geo_embed_state_dict': geo_embed_model.state_dict(),
                'embed_fuse1_state_dict': embed_fuse_model1.state_dict(),
                'embed_fuse2_state_dict': embed_fuse_model2.state_dict(),
                'seq_model_state_dict': seq_model.state_dict(),
                'user_id2idx_dict': user_id2idx_dict, 
                'poi_id2idx_dict': poi_id2idx_dict,
                'cat_id2idx_dict': cat_id2idx_dict,
                'geo_id2idx_dict': geo_id2idx_dict, # 添加 Geohash 字符串到 ID 的映射
                'poi_idx2cat_idx_dict': poi_idx2cat_idx_dict,
                'poi_idx2geohash_id': poi_idx2geohash_id,  # 添加 POI 索引到 Geohash ID 的映射
                'node_attn_map': node_attn_model(X, A), 
                'args': args, 
                'epoch_train_metrics': { 
                    'epoch_train_d_loss': epoch_train_d_loss,
                    'epoch_train_g_loss': epoch_train_g_loss,
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_train_poi_loss': epoch_train_poi_loss,
                    'epoch_train_time_loss': epoch_train_time_loss,
                    'epoch_train_cat_loss': epoch_train_cat_loss,
                    'epoch_train_geo_loss': epoch_train_geo_loss,
                    'epoch_train_top1_acc': epoch_train_top1_acc,
                    'epoch_train_top5_acc': epoch_train_top5_acc,
                    'epoch_train_top10_acc': epoch_train_top10_acc,
                    'epoch_train_top20_acc': epoch_train_top20_acc,
                    'epoch_train_mAP20': epoch_train_mAP20,
                    'epoch_train_ndcg1': epoch_train_ndcg1,
                    'epoch_train_ndcg5': epoch_train_ndcg5,
                    'epoch_train_ndcg10': epoch_train_ndcg10,
                    'epoch_train_ndcg20': epoch_train_ndcg20,
                    'epoch_train_mrr': epoch_train_mrr
                },        
                'epoch_val_metrics': {
                    'epoch_val_d_loss': epoch_val_d_loss,
                    'epoch_val_g_loss': epoch_val_g_loss,
                    'epoch_val_loss': epoch_val_loss,
                    'epoch_val_poi_loss': epoch_val_poi_loss,
                    'epoch_val_time_loss': epoch_val_time_loss,
                    'epoch_val_cat_loss': epoch_val_cat_loss,
                    'epoch_val_geo_loss': epoch_val_geo_loss,
                    'epoch_val_top1_acc': epoch_val_top1_acc,
                    'epoch_val_top5_acc': epoch_val_top5_acc,
                    'epoch_val_top10_acc': epoch_val_top10_acc,
                    'epoch_val_top20_acc': epoch_val_top20_acc,
                    'epoch_val_mAP20': epoch_val_mAP20,
                    'epoch_val_ndcg1': epoch_val_ndcg1,
                    'epoch_val_ndcg5': epoch_val_ndcg5,
                    'epoch_val_ndcg10': epoch_val_ndcg10,
                    'epoch_val_ndcg20': epoch_val_ndcg20,
                    'epoch_val_mrr': epoch_val_mrr
                }
            }
            model_save_dir = os.path.join(args.save_dir, 'checkpoints') #指定模型检查点的保存目录
            # Save best val score epoch
            if monitor_score >= max_val_score: #当前验证评分优于之前保存的最佳评分时才会保存模型
                if not os.path.exists(model_save_dir): os.makedirs(model_save_dir) #检查保存路径是否存在，如果不存在则创建该目录
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt") #将模型的状态字典state_dict保存到文件中
                with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f: #打开/创建一个best_epoch的文本文件，将当前验证周期的验证指标epoch_val_metrics写入文件
                    print(state_dict['epoch_val_metrics'], file=f) 
                max_val_score = monitor_score #将当前的验证评分monitor_score赋值给max_val_score

        # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f: #构建保存文件的完整路径。打开/创建metrics-train.txt
            print(f'train_epochs_d_loss_list={[float(f"{each:.4f}") for each in train_epochs_d_loss_list]}', file=f) #包含所有训练周期损失值的列表
            print(f'train_epochs_g_loss_list={[float(f"{each:.4f}") for each in train_epochs_g_loss_list]}', file=f) #包含所有训练周期损失值的列表
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f) #包含所有训练周期损失值的列表
            print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
            print(f'train_epochs_time_loss_list={[float(f"{each:.4f}") for each in train_epochs_time_loss_list]}',file=f)

            print(f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}', file=f)
            print(f'train_epochs_geo_loss_list={[float(f"{each:.4f}") for each in train_epochs_geo_loss_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f) #top-1准确率
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
            print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                  file=f)
            print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                  file=f)
            print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
            print(f'train_epochs_ndcg1_list={[float(f"{each:.4f}") for each in train_epochs_ndcg1_list]}', file=f)
            print(f'train_epochs_ndcg5_list={[float(f"{each:.4f}") for each in train_epochs_ndcg5_list]}', file=f)
            print(f'train_epochs_ndcg10_list={[float(f"{each:.4f}") for each in train_epochs_ndcg10_list]}', file=f)
            print(f'train_epochs_ndcg20_list={[float(f"{each:.4f}") for each in train_epochs_ndcg20_list]}', file=f)
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_d_loss_list={[float(f"{each:.4f}") for each in val_epochs_d_loss_list]}', file=f)
            print(f'val_epochs_g_loss_list={[float(f"{each:.4f}") for each in val_epochs_g_loss_list]}', file=f)
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
            print(f'val_epochs_time_loss_list={[float(f"{each:.4f}") for each in val_epochs_time_loss_list]}', file=f)
            print(f'val_epochs_cat_loss_list={[float(f"{each:.4f}") for each in val_epochs_cat_loss_list]}', file=f)
            print(f'val_epochs_geo_loss_list={[float(f"{each:.4f}") for each in val_epochs_geo_loss_list]}', file=f)
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
            print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
            print(f'val_epochs_ndcg1_list={[float(f"{each:.4f}") for each in val_epochs_ndcg1_list]}', file=f)
            print(f'val_epochs_ndcg5_list={[float(f"{each:.4f}") for each in val_epochs_ndcg5_list]}', file=f)
            print(f'val_epochs_ndcg10_list={[float(f"{each:.4f}") for each in val_epochs_ndcg10_list]}', file=f)
            print(f'val_epochs_ndcg20_list={[float(f"{each:.4f}") for each in val_epochs_ndcg20_list]}', file=f)
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)

            print(f'last_top1_acc={epoch_val_top1_acc}', file=f)
            print(f'last_top5_acc={epoch_val_top5_acc}', file=f)
            print(f'last_top10_acc={epoch_val_top10_acc}', file=f)
            print(f'last_top20_acc={epoch_val_top20_acc}', file=f)
            print(f'last_mAP20={epoch_val_mAP20}', file=f)
            print(f'last_ndcg1={epoch_val_ndcg1}', file=f)
            print(f'last_ndcg5={epoch_val_ndcg5}', file=f)
            print(f'last_ndcg10={epoch_val_ndcg10}', file=f)
            print(f'last_ndcg20={epoch_val_ndcg20}', file=f)
            print(f'last_mrr={epoch_val_mrr}', file=f)

            print(f'best_top1_acc={best_val_top1_acc}', file=f)
            print(f'best_top5_acc={best_val_top5_acc}', file=f)
            print(f'best_top10_acc={best_val_top10_acc}', file=f)
            print(f'best_top20_acc={best_val_top20_acc}', file=f)
            print(f'best_mAP20={best_val_mAP20}', file=f)
            print(f'best_ndcg1={best_val_ndcg1}', file=f)
            print(f'best_ndcg5={best_val_ndcg5}', file=f)
            print(f'best_ndcg10={best_val_ndcg10}', file=f)
            print(f'best_ndcg20={best_val_ndcg20}', file=f)
            print(f'best_mrr={best_val_mrr}', file=f)
            print(f'best_epoch={best_val_epoch}', file=f)
            print(f'sum_of_train_epochs={epoch}', file=f)

if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    # The name of node features in new_orleans/graph_X.csv
    # The name of node features in NYC1/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    train(args)