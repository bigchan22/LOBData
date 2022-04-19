 #!/bin/bash
python Train_ORD.py --model Trans --data NoPreProcess6feat --lr_initial=1e-4 --num_layers=6 --trainname=6feat6layer-4lr
python Train_ORD.py --model ALSTM --data NoPreProcess6feat --lr_initial=1e-4 --num_layers=6 --trainname=6feat6layer-4lr
python Train_ORD.py --model LSTM --data NoPreProcess6feat --lr_initial=1e-4 --num_layers=6 --trainname=6feat6layer-4lr
