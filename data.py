# """
# load data from datasets according to federated learning requirement
# """
import os
import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset


class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor # rating

        assert self.user_tensor.size(0) == self.item_tensor.size(0) == self.target_tensor.size(0), f"user tensor, item tensor and rating tensor size is not equal, {self.user_tensor.size()} != {self.item_tensor.size()} != {self.target_tensor.size()}"

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    def __init__(self, config):
        self.config = config
        # read file, general dataset
        self.train_ratings, self.test_ratings, self.ratings, self.user_pool, self.item_pool = self._reading()
        # construct negative samples
        self.negatives = self._sample_negative(self.ratings)

    def _reading(self):
        train_file = os.path.join(self.config.data_dir, f"{self.config.data_name}.base")
        test_file = os.path.join(self.config.data_dir, f"{self.config.data_name}.test")

        train_data = pd.read_csv(train_file, sep="\t", header=None, names=["userId", "itemId", "rating", "timestamp"])
        test_data = pd.read_csv(test_file, sep="\t", header=None, names=["userId", "itemId", "rating", "timestamp"])

        # drop timestamp
        train_data = train_data[["userId", "itemId", "rating"]]
        train_data["userId"][:] -= 1
        train_data["itemId"][:] -= 1
        test_data = test_data[["userId", "itemId", "rating"]]
        test_data["userId"][:] -= 1
        test_data["itemId"][:] -= 1

        train_data = self._binarize(train_data)
        test_data = self._binarize(test_data)

        all_data = pd.concat([train_data, test_data])
        user_pool = set(all_data["userId"].unique())
        item_pool = set(all_data["itemId"].unique())

        return train_data, test_data, all_data, user_pool, item_pool

    def _binarize(self, data): # 0 or 1
        data = deepcopy(data)
        data["rating"][data["rating"] > 0] = 1.0
        return data

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]  # negative_samples是evaluation时候用的

    def init_train_data_for_fed_rec(self, num_negatives):
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[["userId", "negative_items"]], on="userId")
        train_ratings["negatives"] = train_ratings["negative_items"].apply(lambda x: random.sample(x, num_negatives))  # 为每个sample随机采样n个negative sample
        rating_lib = {}
        user_pos_neg = {}
        index = 0
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            if row.userId not in rating_lib.keys():
                rating_lib[row.userId] = []
            rating_lib[row.userId].append(index)
            if row.userId not in user_pos_neg.keys():
                user_pos_neg[row.userId] = [[], []]
            user_pos_neg[row.userId][int(row.rating)].append(int(row.itemId))
            index += 1
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))
                rating_lib[row.userId].append(index)
                user_pos_neg[row.userId][0].append(int(row.negatives[i]))
                index += 1

        dataset = [torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(ratings)]
        print(dataset[0].size())
        return dataset, rating_lib, user_pos_neg

    @property
    def evaluate_data(self):
        test_ratings = pd.merge(self.test_ratings, self.negatives[["userId", "negative_samples"]], on="userId")
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]

