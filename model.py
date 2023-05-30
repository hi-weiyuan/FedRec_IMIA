"""recommendation models"""
import torch
import torch.nn as nn
from engine import Engine



class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.latent_dim = config.latent_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()

        for idx, (in_size, out_size) in enumerate(zip(config.layers[:-1], config.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config.layers[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def evaluate_rec(self, u_ids, all_i_ids=None):
        self.eval()
        with torch.no_grad():
            total = len(u_ids)
            print(total)
            num_batch = int(total // 64)
            u_ids_batches = []
            for i in range(num_batch):
                u_ids_batches.append(u_ids[i * 64 : (i + 1) * 64])
            if num_batch * 64 < total:
                u_ids_batches.append(u_ids[num_batch * 64:])
            results = []
            for u_ids in u_ids_batches:
                batch_size = len(u_ids)
                item_embedding =self.embedding_item(all_i_ids) if all_i_ids is not None else self.embedding_item.weight  # [item, n]
                item_total, dim = item_embedding.size()
                user_embedding = self.embedding_user(u_ids)  # [B, n]
                u_e = user_embedding.expand(item_total, batch_size, dim).permute(1, 0, 2)  # [B, i, dim]
                i_e = item_embedding.expand(batch_size, item_total, dim)
                vector = torch.cat([u_e, i_e], dim=-1)
                for idx, _ in enumerate(range(len(self.fc_layers))):
                    vector = self.fc_layers[idx](vector)
                    vector = torch.nn.ReLU()(vector)
                logits = self.affine_output(vector)
                rating = self.logistic(logits) # [B, item, 1]
                results.append(rating)
            rating = torch.cat(results, dim=0)
        return rating


class LightGCN(torch.nn.Module):
    def __init__(self, config):
        super(LightGCN, self).__init__()
        self.config = config
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.config.num_users
        self.num_items = self.config.num_items
        self.latent_dim = self.config.latent_dim
        self.n_layers = self.config.lightGCN_n_layers

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.config.layers[:-1], self.config.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=self.config.layers[-1], out_features=1)

        self.f = nn.Sigmoid()

    def computer(self, g_droped):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def forward(self, users, items, graph):
        # compute embedding
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users]
        items_emb = all_items[items]
        vector = torch.cat([users_emb, items_emb], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.f(logits)

        return rating


class MLPEngine(Engine):
    def __init__(self, config):
        self.model = MLP(config)
        if config.use_cuda:
            self.model.to(config.device_id)
        super(MLPEngine, self).__init__(config)


class LightGCNEngine(Engine):
    def __init__(self, config):
        self.model = LightGCN(config)
        if config.use_cuda:
            self.model.to(config.device_id)
        super(LightGCNEngine, self).__init__(config)
