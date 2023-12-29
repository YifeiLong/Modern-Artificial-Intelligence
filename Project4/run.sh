#!/bin/bash

python GRU.py --embed_size 16 --lr 0.01 --hidden_size 16 --dropout 0.1

python BART.py --lr 0.01 --epoch 20 --batch_size 50
