import pickle
from data import SampleGenerator
import numpy as np

def generate_random_data():
    print("randomly generate training data")
    from run import Arguments
    config = Arguments()
    sample_generator = SampleGenerator(config)
    train_loader, rating_lib, user_pos_neg = sample_generator.init_train_data_for_fed_rec(config.num_negative)
    evaluate_data = sample_generator.evaluate_data

    store_data = {"train_loader": train_loader,
                  "rating_lib": rating_lib,
                  "evaluate_data": evaluate_data,
                  "user_pos_neg": user_pos_neg}
    # print(f"num of forget clients {num_of_forget_clients}")
    print(config.random_data_save_path.format(str(config.num_negative)))
    with open(config.random_data_save_path.format(str(config.num_negative)), "wb") as f:
        pickle.dump(store_data, f)
    print("finish")

generate_random_data()