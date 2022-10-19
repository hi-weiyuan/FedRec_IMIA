import copy
import pickle

import torch
import numpy as np
import time
import pandas as pd
import random
from data import SampleGenerator
from tqdm import tqdm
from utils import fedavg
import math

class Arguments(object):
    def __init__(self):
        # general federated settings
        self.data_name = "ml-100k"
        self.data_dir = "dataset/ml-100k" # note
        self.graph_path = "graphs/ml-100k.pkl"
        self.random_data_save_path = "dataset/ml-100k/processed/neg_sample_{}.pkl" # note
        self.model_type = "NCF"
        self.model_dir = "save/" + self.model_type +  "_HR{:.4f}_NDCG{:.4f}.model"   # model save place # note
        self.global_epoch = 200  # global rounds of training
        self.local_epoch = 20  # local training epoch # note
        self.use_cuda = torch.cuda.is_available()
        self.device_id = "cuda:1"  # note
        self.num_users = 943  # ml-100k user number-943
        self.num_items = 1682 # ml-100k item numer-1682
        self.lightGCN_n_layers = 1
        self.layers = [128, 64, 32]
        self.num_negative = 4   # pos:neg = 1:4
        self.latent_dim = 64  # embedding dim
        self.client_frac = 0.01  # fraction of clients who attend training
        self.local_bs = 64
        self.lr = 0.001
        self.optimizer = "adam"
        self.seed = 42   # one of my favorite num of seeds haha

        self.save_all_model = True
        self.train_with_test = True

        self.need_fed_train = False

        # defend method
        self.defend_method = "none"   # ldp, user_enhanced, None
        self.std = 0.001  # note: noise scale, for ldp
        self.mu = 0.7  # note: factor to control regularization of item updates.


def fl_train(engine, train_loader, rating_lib, evaluate_data, config):
    print(5 * "#" + "  Federated training Start  " + 5 * "#")
    all_global_models = list()
    all_client_models = list()
    all_hits_ndcg = list()
    all_client_items = list()
    all_client_train_data = list()
    global_model = engine.model

    all_global_models.append(copy.deepcopy(global_model.cpu()))
    test_time_cost = 0.

    for epoch in tqdm(range(config.global_epoch)):
        print("\n")
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        num_of_training_client = max(int(config.client_frac * config.num_users), 1)
        all_clients = [i for i in range(config.num_users)]

        idxs_users = np.array([x for x in np.random.choice(all_clients, num_of_training_client, replace=False)])

        global_model, client_models, local_losses, local_items, local_train_data = engine.fed_train_an_global_epoch(train_loader, rating_lib, idxs_users, epoch, config.device_id)
        all_global_models.append(copy.deepcopy(global_model))
        all_client_models.append(client_models)
        all_client_items.append(local_items)

        print(f"Epoch {epoch} ends ! the average loss is {sum(local_losses) / len(local_losses)}")
        engine.model = global_model

        if config.train_with_test:
            print(f"Epoch {epoch} evaluate start!")
            s = time.time()
            hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch, device_id=config.device_id)
            e = time.time()
            test_time_cost += (e - s)
            all_hits_ndcg.append((hit_ratio, ndcg))
            print(f"Epoch {epoch}'s performance: hit@10: {hit_ratio}, ndcg@10: {ndcg}, evaluate cost {e-s} time")

    if not config.train_with_test:
        s = time.time()
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=config.global_epoch-1, device_id=config.device_id)
        print(f"Epoch {config.global_epoch-1}'s performance: hit@10: {hit_ratio}, ndcg@10: {ndcg}")
        e = time.time()
        test_time_cost += (e - s)
        all_hits_ndcg.append((hit_ratio, ndcg))
    print(5 * "#" + "  Federated training End  " + 5 * "#")
    return all_global_models, all_client_models, test_time_cost, all_hits_ndcg, all_client_items, all_client_train_data

def run():
    # step1: set parameters, set random seeds
    config = Arguments()
    print(config.__dict__)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # step2: construct data
    if config.model_type == "LightGCN":
        from model import LightGCNEngine
        engine = LightGCNEngine(config)
    else:
        from model import MLPEngine
        engine = MLPEngine(config)

    print(config.random_data_save_path.format(str(config.num_negative)))
    with open(config.random_data_save_path.format(str(config.num_negative)), "rb") as f:
        stored_data = pickle.load(f)
        train_loader = stored_data["train_loader"]
        rating_lib = stored_data["rating_lib"]
        evaluate_data = stored_data["evaluate_data"]
        user_pos_neg = stored_data["user_pos_neg"]

    if config.model_type == "LightGCN":
        engine._init_graph(train_loader, rating_lib)

    # step 3: train
    start_time = time.time()
    all_global_models, all_client_models, test_cost_time, hits_ndcgs, all_client_items, all_client_train_data = fl_train(engine, train_loader, rating_lib, evaluate_data, config)
    end_time = time.time()
    print(f"Fed Learning time consuming = {end_time - start_time - test_cost_time}, test time cost is {test_cost_time} ")
    if config.save_all_model:
        save_dict = {"all_global_models": all_global_models,
                 "all_client_models": all_client_models,
                 "all_hits_ndcg":hits_ndcgs,
                 "all_client_items": all_client_items
                 }
        torch.save(save_dict, config.model_dir.format(hits_ndcgs[-1][0], hits_ndcgs[-1][1]))

if __name__ == '__main__':
    run()


