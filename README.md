# FedRec_IMIA
Implementation for the paper [Interaction-level Membership Inference Attack Against Federated Recommender Systems](https://arxiv.org/abs/2301.10964).

# How to use
*To convenient the experiment, all clients' models are stored during training process.
This setting can avoid retraining FedRecs when searching hyper-parameters, however, it will cost some storage.
If your computer does not have enough storage, you can rewrite the code to run FedRec and IMIA simultaneously, or randomly store a part of clients.*

- Step 1. put your dataset in corresponding directory (e.g. put ml-100k.base and ml-100k.test in dataset/ml-100k/).
- Step 2. set hyper-parameters in Argument class in run.py.
- Step 3. run run.py to train FedRec.
- Step 4. set hyper-parameters for IMIA in Config class in membership_attack.py
- Step 5. run membership.py.

# Citation
**Please cite the paper if the code is helpful. Thanks!**
```
@inproceedings{yuan2023interaction,
  title={Interaction-level Membership Inference Attack Against Federated Recommender Systems},
  author={Yuan, Wei and Yang, Chaoqun and Nguyen, Quoc Viet Hung and Cui, Lizhen and He, Tieke and Yin, Hongzhi},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={1053--1062},
  year={2023}
}
```