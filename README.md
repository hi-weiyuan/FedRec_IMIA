# FedRec_IMIA
Implementation for Interaction-level Membership Inference Attack Against Federated Recommender Systems.

# How to use
Step 1. put your dataset (already split by leave-one-out) in corresponding directory (e.g. put ml-100k.base and ml-100k.test in dataset/ml-100k/).
Step 2. create a directory "processed" under above data's directory. (e.g. dataset/ml-100k/processed/)
Step 3. Revise Argument class in run.py.
Step 4. Run data_preprocess.py.
Step 5. Run run.py.
Step 6. Run membership.py.
