 #!/bin/bash
python -u Train_ORD.py --model Trans --device cuda:0 --data _10_0823Train  --lr_initial 1e-3  --num_layers 4 --batch_size 64 --trainname 0824Train
python -u Train_ORD.py --model ALSTM --device cuda:0 --data _10_0823Train  --lr_initial 1e-3  --num_layers 4 --batch_size 64 --trainname 0824Train
python -u Train_ORD.py --model LSTM --device cuda:0 --data _10_0823Train --lr_initial 1e-3  --num_layers 4 --batch_size 64 --trainname 0824Train
