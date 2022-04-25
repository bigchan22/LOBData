 #!/bin/bash
python Train_ORD.py --model Trans --device cuda:1 --data 6feat --lr_initial=1e-4 --num_layers=8 --trainname=6feat8layer-4lr
#python Train_ORD.py --model ALSTM --data 6feat --lr_initial=1e-3 --num_layers=8 --trainname=6feat8layer-3lr
#python Train_ORD.py --model LSTM --data 6feat --lr_initial=1e-3 --num_layers=8 --trainname=6feat8layer-3lr
