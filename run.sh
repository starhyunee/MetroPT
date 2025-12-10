#!/bin/bash

PYTHON_FILE="main.py"
# MODEL="MAD_GAN"
# DATASET="UCR"
# python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --retrain
#MODEL="Transformer_attention"
#DATASET="MetroPT"
#batch_size = batch_size
#python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --retrain --batch_size 64 --total_epochs 100
#python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --test --batch_size 64 --total_epochs 100



MODEL="Transformer_layer2"
DATASET="MetroPT"
# #batch_size = batch_size
python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --test --batch_size 64 --total_epochs 100

# MODEL="TranAD_multiview"
# DATASET="MetroPT"
# #batch_size = batch_size
# python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --retrain --batch_size 64 --total_epochs 30






# # PYTHON_FILE="main.py"
# MODEL="TranAD_multiview_temp2"
# DATASET="UCR"
# python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --retrain
# #DATASET="PSM"
# #python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --retrain

# DATASET="SMD"
# python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --retrain --batch_size 64

# DATASET="SMAP"
# python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --retrain --batch_size 64

# DATASET="MSL" 
# python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --retrain --batch_size 64

# DATASET="SWaT" 
# python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --retrain --batch_size 64

# DATASET="NIPS_TS_Swan" 
# python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --retrain

#DATASET="NIPS_TS_Water" 
#python3 $PYTHON_FILE --model $MODEL --dataset $DATASET --retrain


