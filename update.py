import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import functools
import copy
from data import UserItemRatingDataset
from utils import use_optimizer
from torch.autograd import Variable


def variable(t: torch.Tensor, device, **kwargs):

    t = t.to(device)
    return Variable(t, **kwargs)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        users, items, ratings = self.dataset[self.idxs][0],self.dataset[self.idxs][1], self.dataset[self.idxs][2]
        return users[item], items[item], ratings[item]


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, userid, graph=None):
        self.args = args
        # self.logger = logger
        self.userid = userid
        users, items, ratings = dataset
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        self.trainloader = self.train_val_test(dataset, idxs)
        self.device = args.device_id
        self.criterion = torch.nn.BCELoss()
        self.graph = graph


    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(1.0 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        return trainloader

    def train_single_batch(self, users, items, ratings, optimizer, model, previous_item_embedding=None):
        if self.args.use_cuda is True:
            users, items, ratings = users.to(self.args.device_id), items.to(self.args.device_id), ratings.to(self.args.device_id)
        optimizer.zero_grad()

        if self.graph == None:
            ratings_pred = model(users, items)
        else:
            ratings_pred = model(users, items, self.graph)

        loss = self.criterion(ratings_pred.view(-1), ratings)
        if previous_item_embedding != None:
            for n, p in model.named_parameters():
                # print(n)
                if n == "embedding_item.weight":
                    _loss = ((p - previous_item_embedding) ** 2).sum()
                    loss += self.args.mu * _loss
        loss.backward()
        optimizer.step()
        loss = loss.item()
        if self.args.defend_method == "user_enhanced":
            return loss, _loss
        else:
            return loss

    def update_weights(self, model, global_ground, defend_method="ldp", grad_limit=1., std=1e-6):
        received_model_dict = copy.deepcopy(model.state_dict())
        model_dict = model.state_dict()
        user_embedding_set = torch.tensor(np.zeros((self.args.num_users, self.args.latent_dim)))
        user_embedding_set[self.userid] = model_dict['embedding_user.weight'][self.userid]
        state_dict = {'embedding_user.weight': user_embedding_set}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        model.train()

        epoch_loss = []
        epoch_regular = []

        optimizer = use_optimizer(model, self.args)
        epochs = self.args.local_epoch
        all_local_items = []
        if self.args.defend_method == "user_enhanced":
            previous_item_embedding = variable(copy.deepcopy(model.state_dict()['embedding_item.weight'].data),
                                               self.args.device_id)
        for iter in range(epochs):
            batch_loss = []
            batch_regular = []
            for batch_id, batch in enumerate(self.trainloader):
                user, item, rating = batch[0], batch[1], batch[2]
                all_local_items.extend(item.tolist())
                rating = rating.float()
                if self.args.defend_method == "user_enhanced":
                    loss, regular = self.train_single_batch(user, item, rating, optimizer, model, previous_item_embedding)
                    batch_regular.append(regular.item())
                else:
                    loss = self.train_single_batch(user, item, rating, optimizer, model)
                batch_loss.append(loss)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if len(batch_regular) != 0:
                epoch_regular.append(sum(batch_regular) / len(batch_regular))
        all_local_items = list(set(all_local_items))

        if defend_method == "ldp":
            current_model_dict = copy.deepcopy(model.state_dict())
            old_model_dict = copy.deepcopy(received_model_dict)
            for layer in current_model_dict.keys():
                if layer != "embedding_user.weight":
                    grad = current_model_dict[layer] - received_model_dict[layer]
                    if layer == "embedding_item.weight":
                        grad = grad[torch.Tensor(all_local_items).long()]
                        grad_norm = grad.norm(2, dim=-1, keepdim=True)
                        grad_max = grad_limit
                        too_large = grad_norm[:, 0] > grad_max
                        grad[too_large] /= (grad_norm[too_large] / grad_max)
                        grad += self.noise(grad.shape, std)
                        old_model_dict[layer][torch.Tensor(all_local_items).long()] += grad
                    else:
                        need_squeeze = False
                        if len(grad.size()) == 1:
                            grad = grad.unsqueeze(-1)
                            need_squeeze = True
                        grad_norm = grad.norm(2, dim=-1, keepdim=True)
                        grad_max = grad_limit
                        too_large = grad_norm[:, 0] > grad_max
                        grad[too_large] /= (grad_norm[too_large] / grad_max)
                        grad += self.noise(grad.shape, std)
                        if need_squeeze:
                            grad = grad.squeeze()
                        old_model_dict[layer] += grad
            old_model_dict.update({"embedding_user.weight": current_model_dict["embedding_user.weight"]})
            model.load_state_dict(old_model_dict)

        if len(epoch_regular) != 0 :
            print(f"epoch loss is {sum(epoch_loss) / len(epoch_loss)}, regular loss is {sum(epoch_regular) / len(epoch_regular)}")

        return model, sum(epoch_loss) / len(epoch_loss), model.state_dict()['embedding_user.weight'][self.userid], all_local_items


    def noise(self, shape, std):
        noise = np.random.multivariate_normal(
            mean=np.zeros(shape[1]), cov=np.eye(shape[1]) * std, size=shape[0]
        )
        return torch.Tensor(noise).to(self.args.device_id)