1. The functions named run_xxx_main in main.py are training and testing functions for four datasets respectively.
2. The definition of parameters is in main.py
3. Run the shell scripts named run_xxx.sh to test the results
3. The original dataset path is ./data/raw_data
4. The four preprocessed dataset paths are ./data/bundle_time, ./data/ele_time, ./data/movielens1m, ./data/taobao
5. Preprocessed dataset includes lbes.pkl, mms.pkl, train_data.csv, val_data.csv, and test_data.csv
### Open Tensorboard Monitor
```
tensorboard --logdir logs/xxx_logs --host localhost --port 0000
```
