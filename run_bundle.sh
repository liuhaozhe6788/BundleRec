# python main.py --dataset bundle --batch_size 512 --lr 1e-3 --transformer_num 1  --embedding 8 --num_head 8 --seed 0 --data_path ./data/bundle_time --max_len 8 --force_restart  > "bundle_train.log" 2>&1

python main.py --dataset bundle --batch_size 512 --lr 1e-3 --transformer_num 1  --embedding 8 --num_head 8 --seed 0 --data_path ./data/bundle_time --max_len 8   > "bundle_test.log" 2>&1