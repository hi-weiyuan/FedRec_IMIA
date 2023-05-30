import torch
import copy


def use_optimizer(network, params):
    if params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=params.lr)
    elif params.optimizer == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=params.lr)
    elif params.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params.lr,
                                        alpha=params.rmsprop_alpha,
                                        momentum=params.rmsprop_momentum)
    else:
        raise ImportError(f"we don't implement {params.optimizer}, please use [sgd, adam, rmsprop]")
    return optimizer


def fedavg(local_models):
    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()

    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())

    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer]
        avg_state_dict[layer] /= len(local_models)

    global_model.load_state_dict(avg_state_dict)
    return global_model
