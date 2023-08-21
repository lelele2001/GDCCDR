# GDCCDR
The source code is official pytorch implementation of GDCCDR (Graph Disentangled Contrastive Learning with Personalized Transfer for Cross-Domain Recommendation) by Jing Liu, Lele Sun, Weizhi Nie, PeiGuang Jing and Yuting Su.
# Requirements:
Python == 3.8.13
PyTorch == 1.11.0
torch-sparse == 0.6.13
numpy == 1.22.3
# Datasets
We use four Amazon datasets (Sport&Phone, Sport&Cloth, Elec&Phone, Elec&Cloth) to evaluate our GDCCDR and preprocess these datasets following BITGCF and DisenCDR.
# Training
You can use these commands to train the model:
python main.py --dataset sport_phone --ecl_reg 0.2 --pcl_reg 0.001 --alpha 0.25 --beta 0.03 --layer 6
python main.py --dataset sport_cloth --ecl_reg 0.05 --pcl_reg 0.05 --alpha 0.1 --beta 0.3 --layer 5
