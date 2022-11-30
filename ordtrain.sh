 #!/bin/bash
python -u Train_ORD.py --model Trans --device cuda:0 --data 1111Train_08  --lr_initial 1e-3  --num_layers 4 --batch_size 64 --trainname 1111T
python -u Train_ORD.py --model ALSTM --device cuda:0 --data 1111Train_08  --lr_initial 1e-3  --num_layers 4 --batch_size 64 --trainname 1111T
python -u Train_ORD.py --model LSTM --device cuda:0 --data 1111Train_08 --lr_initial 1e-3  --num_layers 4 --batch_size 64 --trainname 1111T
