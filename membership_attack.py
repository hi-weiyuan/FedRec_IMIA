
import math
import random

import torch
from run import Arguments
import pickle
import copy

from tqdm import tqdm

class Config(object):
    def __init__(self):
        self.model_path = "./save/XXXXX.model"
        self.data_path = "dataset/steam-200k/processed/neg_sample_4.pkl"

        self.unknown_user_init_method = True
        self.same_distribution = False
        self.start_from_init = True
        self.use_popularity = False
        self.lam = 0.9

        # original fed learning hyper-parameters
        self.select_ratio = 0.2  # 1 / (1 + 4)
        self.batch_size = 64
        self.shuffle = True
        self.lr = 0.001
        self.local_epoch = 20
        self.device_type = "cuda:3"

        self.model_type = "LightGCN"

        print(self.__dict__)
        with open("dataset/steam-200k/processed/neg_sample_4.pkl", "rb") as f:
            self.stored_data = pickle.load(f)
            self.train_loader = self.stored_data["train_loader"]
            self.rating_lib = self.stored_data["rating_lib"]
            self.evaluate_data = self.stored_data["evaluate_data"]
            self.user_pos_neg = self.stored_data["user_pos_neg"]
        if self.model_type == "LightGCN":
            with open("./graphs/steam-200k.pkl", "rb") as f:
                self.graphs = pickle.load(f)
            self.graphs = {k: v.to(self.device_type) for k, v in self.graphs.items()}

def IMIA(config):
    fed_data_dict = torch.load(config.model_path)
    global_model = fed_data_dict["all_global_models"]
    client_models = fed_data_dict["all_client_models"]
    client_items = fed_data_dict["all_client_items"]
    plist = []
    rlist = []
    flist = []

    random_plist = []
    random_rlist = []
    random_flist = []

    if not config.start_from_init:
        reversed_client_models = copy.deepcopy(client_models[::-1])
    for u_id in tqdm(config.user_pos_neg.keys()):
        idx = None
        if config.start_from_init:
            for i, c in enumerate(client_models):
                if u_id in c.keys():
                    print(i)
                    idx = i
                    break
        else:
            for i, c in enumerate(reversed_client_models):
                if u_id in c.keys():
                    idx = len(reversed_client_models) - i - 1
                    print(idx)
                    break

        if idx is None:
            continue
        g = copy.deepcopy(global_model[idx])
        if config.unknown_user_init_method:
            if config.same_distribution:
                torch.nn.init.normal_(g.embedding_user.weight)

            else:
                torch.nn.init.xavier_normal_(g.embedding_user.weight)

        cm = client_models[idx]
        ci = client_items[idx]
        p, r, f = IMIA_for_one_user(config, u_id, config.user_pos_neg, g, cm, ci)
        plist.append(p)
        rlist.append(r)
        flist.append(f)
        neg, pos = config.user_pos_neg[u_id]
        all_item = list(set(pos)) + list(set(neg))
        random.shuffle(all_item)
        random_select = all_item[:int(len(all_item) * config.select_ratio)]
        print(f"user {u_id}, find at epoch {idx}, pos length {len(pos)}, select length {len(random_select)}, the p is {p}, the r is {r}, and the f is {f}")
        p = len(set(random_select).intersection(set(pos))) / len(random_select)
        # print(f"pos length {len(pos)}, select length {len(random_select)}, all item length {len(all_item)}")
        r = len(set(random_select).intersection(set(pos))) / len(pos)
        if p + r == 0:
            f = 0
        else:
            f = 2 * p * r / (p + r)
        random_plist.append(p)
        random_rlist.append(r)
        random_flist.append(f)

    print(f"we attacked {len(plist)} users, the following is performance result:")
    print(f"performance: p: {sum(plist) / len(plist)}, r: {sum(rlist) / len(rlist)}, f1: {sum(flist) / len(flist)}")
    print(f"random performance: p: {sum(random_plist) / len(random_plist)}, r: {sum(random_rlist) / len(random_rlist)}, f1: {sum(random_flist) / len(random_flist)}")

def IMIA_for_one_user(config, user_id, user_pos_neg, global_model, client_models, client_items):
    # print(client_models.keys())
    neg, pos = user_pos_neg[user_id]
    all_items = list(set(pos)) + list(set(neg))
    confirmed_pos = []
    confirmed_neg = []
    un_confirmed = copy.deepcopy(all_items)
    random.shuffle(un_confirmed)
    random_selected = un_confirmed[:int(config.select_ratio * len(un_confirmed))]
    no_random_selected = un_confirmed[int(config.select_ratio * len(un_confirmed)):]
    while len(confirmed_pos) < int(config.select_ratio * len(all_items)) or len(un_confirmed) > int(config.select_ratio * len(all_items)):
        users = [user_id] * len(all_items)
        # print(f"users len: {len(users)}, all item len {len(all_items)}, set all item len {len(list(set(all_items)))}")
        new_items = random_selected + no_random_selected
        new_ratings = [1] * len(random_selected) + [0] * len(no_random_selected)
        idxes = [i for i in range(len(users))]
        model = copy.deepcopy(global_model)

        model = train_model(config, users, new_items, new_ratings, idxes, model)
        old_item_embedding = copy.deepcopy(client_models[user_id].state_dict()["embedding_item.weight"])
        old_item_set_embedding = []
        for idx in client_items[user_id]:
            old_item_set_embedding.append(old_item_embedding[idx].squeeze())
        old_item_set_embedding = torch.stack(old_item_set_embedding)
        current_item_embedding = copy.deepcopy(model.state_dict()["embedding_item.weight"])
        current_item_set_embedding = []
        for idx in client_items[user_id]:
            current_item_set_embedding.append(current_item_embedding[idx].squeeze())
        current_item_set_embedding = torch.stack(current_item_set_embedding)
        difference = torch.nn.functional.pairwise_distance(old_item_set_embedding, current_item_set_embedding)
        indexes = torch.argsort(difference, descending=False).tolist()
        num = 0
        for i in indexes:
            item = client_items[user_id][i]
            if item not in confirmed_neg and item not in confirmed_pos:
                if item in random_selected:
                    confirmed_pos.append(item)
                else:
                    confirmed_neg.append(item)
                num += 1

            if num > int(config.lam * len(all_items)):
                break
        un_confirmed = list(set(all_items) - set(confirmed_neg + confirmed_pos))
        random.shuffle(un_confirmed)
        expect_pos_num = int(config.select_ratio * len(all_items))
        if len(confirmed_pos) >= expect_pos_num:
            break
        random_selected = confirmed_pos + un_confirmed[:expect_pos_num - len(confirmed_pos)]
        no_random_selected = confirmed_neg + un_confirmed[expect_pos_num - len(confirmed_pos):]
    p = len(set(confirmed_pos).intersection(set(pos))) / len(confirmed_pos)
    r = len(set(confirmed_pos).intersection(set(pos))) / len(pos)
    if p + r == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)

    return p, r, f

def train_model(config, users, new_items, new_ratings, idxes, model):
    graph = None
    if config.model_type == "LightGCN":
        graph = copy.deepcopy(config.graphs[users[0]])
    users, items, ratings = torch.LongTensor(users), torch.LongTensor(new_items), torch.FloatTensor(new_ratings)
    from data import UserItemRatingDataset
    from update import DatasetSplit
    from torch.utils.data import DataLoader, Dataset
    from utils import use_optimizer
    dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                    item_tensor=torch.LongTensor(items),
                                    target_tensor=torch.FloatTensor(ratings))
    train_loader = DataLoader(DatasetSplit(dataset, idxes), batch_size=config.batch_size, shuffle=config.shuffle)
    criterion = torch.nn.BCELoss()
    model = model.to(config.device_type)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for iter in range(config.local_epoch):
        for batch_id, batch in enumerate(train_loader):
            user, item, rating = batch[0].to(config.device_type), batch[1].to(config.device_type), batch[2].to(config.device_type)
            rating = rating.float()
            if graph is None:
                ratings_pred = model(user, item)
            else:
                ratings_pred = model(user, item, graph)
            loss = criterion(ratings_pred.view(-1), rating)
            loss.backward()
            optimizer.step()
    return model.cpu()


if __name__ == '__main__':
    random.seed(42)
    config = Config()
    IMIA(config)
