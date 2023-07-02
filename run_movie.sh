# python main.py --dataset movie --batch_size 512 --lr 1e-3 --transformer_num 1 --embedding 8  --num_head 8 --seed 0 --data_path ./data/movielens1m --max_len 8  --force_restart  > "movie_train.log" 2>&1

python main.py --dataset movie --batch_size 512 --lr 1e-3 --transformer_num 1 --embedding 8 --num_head 8 --seed 0 --data_path ./data/movielens1m --max_len 8  > "movie_test.log" 2>&1