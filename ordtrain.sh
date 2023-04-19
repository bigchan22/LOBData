 #!/bin/bash
#python -u Train_ORD.py --model Trans --device cuda:0 --data 0307Trainset  --lr_initial 1e-3  --num_layers 6 --batch_size 64 --trainname 0311T
python -u Train_ORD.py --model ALSTM --device cuda:0 --data 0307Trainset  --lr_initial 1e-3  --num_layers 6 --batch_size 64 --trainname 0311T
python -u Train_ORD.py --model LSTM --device cuda:0 --data 0307Trainset --lr_initial 1e-3  --num_layers 6 --batch_size 64 --trainname 0311T
