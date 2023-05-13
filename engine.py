import copy
import pickle
import time

import numpy as np
import pandas as pd
import torch
import os

from utils import use_optimizer, fedavg
from update import LocalUpdate
from metrics import Metrics
from data import SampleGenerator
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from tqdm import tqdm


class Engine(object):
    def __init__(self, config):
        self.config = config
        self._metrics = Metrics(top_k=10)
        self.optimizer = use_optimizer(self.model, config)

        self.criterion = torch.nn.BCELoss()
        self.global_weights_collection = []
        s_g = SampleGenerator(config)
        self.negatives = s_g.negatives
        self.graphs = {i:None for i in range(self.config.num_users)}

    def _get_graph(self, userList, itemList):
        def _convert_sp_mat_to_sp_tensor(X):
            coo = X.tocoo().astype(np.float32)
            row = torch.Tensor(coo.row).long()
            col = torch.Tensor(coo.col).long()
            index = torch.stack([row, col])
            data = torch.FloatTensor(coo.data)
            return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        trainUser = np.array(userList)
        trainItem = np.array(itemList)
        adj_mat = sp.dok_matrix((self.config.num_users + self.config.num_items, self.config.num_users + self.config.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        UserItemNet = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)),
                   shape=(self.config.num_users, self.config.num_items))
        R = UserItemNet.tolil()
        adj_mat[:self.config.num_users, self.config.num_users:] = R
        adj_mat[self.config.num_users:, :self.config.num_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        graph = _convert_sp_mat_to_sp_tensor(norm_adj)
        graph = graph.coalesce().to(self.config.device_id)
        return graph

    def get_graph(self, old_train_loader, old_rating_lib):
        users, items, ratings = old_train_loader
        users = users.tolist()
        items = items.tolist()
        ratings = ratings.tolist()

        users = [users[i] for i in old_rating_lib]
        items = [items[i] for i in old_rating_lib]
        ratings = [ratings[i] for i in old_rating_lib]

        pos_u = []
        pos_i = []
        for idx, r in enumerate(ratings):
            if r == 1:
                pos_u.append(users[idx])
                pos_i.append(items[idx])

        graph = self._get_graph(copy.deepcopy(pos_u), copy.deepcopy(pos_i))
        return graph

    def _init_graph(self, train_loader, rating_lib):
        if os.path.exists(self.config.graph_path):
            print(f"already has graphs, load it!")
            with open(self.config.graph_path, "rb") as f:
                self.graphs = pickle.load(f)
            self.graphs = {k: v.to(self.config.device_id) for k, v in self.graphs.items()}
            print(f"directly load graphs from {self.config.graph_path}")
        else:
            u_ids = [i for i in range(self.config.num_users)]
            self.graphs = {}
            for idx in tqdm(u_ids):  # generate local bipartite graph
                self.graphs[idx] = self.get_graph(train_loader, rating_lib[idx])
            tmp = copy.deepcopy(self.graphs)
            tmp = {k: v.to("cpu") for k, v in tmp.items()}
            with open(self.config.graph_path, "wb") as f:
                pickle.dump(tmp, f)
            print(f"finish init graphs and save at {self.config.graph_path}")

    def fed_train_an_global_epoch(self, train_loader, rating_lib, idxs_users, epoch_id, device_id):
        self.model = self.model.to(device_id)
        self.model.train()

        local_losses = []
        clients_model_and_ids = {}
        clients_local_items_and_ids = {}
        clients_train_data_and_ids = {}
        user_embedding_lib = self.model.state_dict()['embedding_user.weight']
        for idx in idxs_users:
            idx_rating_lib = rating_lib[idx]
            data_loader = train_loader  # 1：4
            local_model = LocalUpdate(args=self.config, dataset=data_loader, idxs=idx_rating_lib, userid=idx, graph=copy.deepcopy(self.graphs[idx]))
            w, loss, user_em, local_items = local_model.update_weights(model=copy.deepcopy(self.model), global_ground=epoch_id, defend_method=self.config.defend_method, std=self.config.std)
            clients_model_and_ids[idx] = copy.deepcopy(w.cpu())
            clients_local_items_and_ids[idx] = local_items
            local_losses.append(copy.deepcopy(loss))
            user_embedding_lib[idx] = user_em

        # fed avg
        global_weights = fedavg(list(clients_model_and_ids.values()))

        state_dict = {'embedding_user.weight': user_embedding_lib}
        model_dict = global_weights.state_dict()
        model_dict.update(state_dict)
        global_weights.load_state_dict(model_dict)
        return global_weights, clients_model_and_ids, local_losses, clients_local_items_and_ids, clients_train_data_and_ids

    def evaluate(self, evaluate_data, epoch_id, device_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model = self.model.to(device_id)
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            if self.config.use_cuda:
                test_users = test_users.to(device_id)
                test_items = test_items.to(device_id)
                negative_users = negative_users.to(device_id)
                negative_items = negative_items.to(device_id)

            if self.config.model_type == "LightGCN":
                test_scores = []

                for i in range(test_users.size(0)):
                    test_s = self.model(test_users[i], test_items[i], self.graphs[test_users[i].item()]).cpu()
                    test_scores.append(test_s)

                negative_scores = []
                negative_users_list = negative_users.data.tolist()
                negative_items_list = negative_items.data.tolist()
                total_num = int(len(negative_users_list) / 99)
                for i in range(total_num):
                    u_list = negative_users_list[i * 99: (i + 1) * 99]
                    i_list = negative_items_list[i * 99: (i + 1) * 99]
                    neg_s = self.model(torch.LongTensor(u_list).to(device_id), torch.LongTensor(i_list).to(device_id), self.graphs[u_list[0]]).squeeze().cpu()
                    negative_scores.append(neg_s)
                test_scores = torch.stack(test_scores)
                negative_scores = torch.cat(negative_scores)
            else:
                test_scores = self.model(test_users, test_items)
                negative_scores = self.model(negative_users, negative_items)
            self._metrics.subjects = [test_users.data.view(-1).tolist(),
                                     test_items.data.view(-1).tolist(),
                                     test_scores.data.view(-1).tolist(),
                                     negative_users.data.view(-1).tolist(),
                                     negative_items.data.view(-1).tolist(),
                                     negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metrics.cal_hit_ratio(), self._metrics.cal_ndcg()
        print('[Evluating Epoch {}] HR = {:.7f}, NDCG = {:.7f}'.format(epoch_id, hit_ratio, ndcg))
        return hit_ratio, ndcg