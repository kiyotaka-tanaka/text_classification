# text_classification


Requirements
tensorflow 0.12
使い方：

python train.py --vocab_file "data_kindle.txt"  --rnn_size 512 --embedding_size 50 --n_classes 2 --epoch_size 500


python predict.py --vocab_file "data_kindle.txt" --input "今回漫画モデルとページめくりも高速になってるということと丁度セールの対象だったので購入してみました。" --rnn_size 512 --embedding_size 50 --n_classes 2
