 #!/bin/bash
python Train_AttentionLSTM_ORD.py --lr_initial=1e-3 --num_layers=6
python Train_LSTM_ORD.py --lr_initial=1e-3 --num_layers=6
python Train_Transformer_ORD.py --lr_initial=1e-3 --num_layers=6
