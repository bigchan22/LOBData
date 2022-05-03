 #!/bin/bash
#python Train_ORD.py --model Trans --device cuda:0 --data norm6feat --lr_initial 1e-3  --num_layers 4 --batch_size 64 --trainname norm6feat6layer-3lrb64
#python Train_ORD.py --model ALSTM --device cuda:0 --data norm6feat --lr_initial 1e-3  --num_layers 4 --batch_size 64 --trainname norm6feat6layer-3lrb64
python Train_ORD.py --model LSTM --device cuda:1 --data norm6feat --lr_initial 1e-3  --num_layers 4 --batch_size 64 --trainname norm6feat6layer-3lrb64
