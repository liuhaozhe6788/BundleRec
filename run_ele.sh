python main.py --dataset ele --batch_size 512 --lr 1e-3 --transformer_num 2 --embedding 8 --num_head 8 --seed 0 --data_path ./data/ele_time --max_len 51 --force_restart  > "ele_train.log" 2>&1
python main.py --dataset ele --batch_size 512 --lr 1e-3 --transformer_num 2 --embedding 8 --num_head 8 --seed 0 --data_path ./data/ele_time --max_len 51  > "ele_test.log" 2>&1

