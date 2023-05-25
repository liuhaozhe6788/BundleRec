import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from data.bundle_dataset import BundleDataset
from data.ele_dataset import EleDataset
from data.movie_dataset import MovieDataset
from model.BST import BST
from utils import find_ckpt_file


def bundle_main(args, v_num, train_dataset, val_dataset, test_dataset):
    cat_features = ['uid', 'is_visitor', 'gender', 'is_up_to_date_version', 'register_country', 'register_device',
                      'register_device_brand', 'register_os', 'bundle_id']
    num_features = ['register_time', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                      'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                      'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                      'life_time', 'star', 'battle_pass_exp', 'pet_heart_stock', 'pet_exp_stock', 'friend_cnt',
                      'social_friend_cnt', 'pay_amt', 'pay_cnt', 'pay_per_day', 'pay_mean']
    dnn_col = ['uid', 'register_country', 'register_time', 'is_visitor', 'register_device',
               'register_device_brand', 'register_os', 'gender']
    transformer_col = ['bundle_id', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                       'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                       'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                       'life_time', 'star', 'battle_pass_exp', 'is_up_to_date_version', 'pet_heart_stock',
                       'pet_exp_stock', 'friend_cnt', 'social_friend_cnt', 'pay_amt', 'pay_cnt',
                       'pay_per_day', 'pay_mean']
    target_col = 'impression_result'
    time_col = 'ftime'
    pl.seed_everything(args.seed)
    model = BST(cat_features=cat_features,
                num_features=num_features,
                dnn_col=dnn_col,
                transformer_col=transformer_col,
                target_col=target_col,
                time_col=time_col,
                args=args)
    logger = TensorBoardLogger(save_dir='logs/bundle_logs/train' if args.force_restart else 'logs/bundle_logs/test',
                               name='bundle_BST_final',
                               version=v_num)
    callback = ModelCheckpoint(monitor="val/auc",
                               mode="max",
                               save_top_k=1,
                               filename='epoch={epoch}-step={step}-val_auc={val/auc:.4f}-log_loss={val\loss:.4f}',
                               save_weights_only=True,
                               auto_insert_metric_name=False)
    callback2 = EarlyStopping(monitor="val/auc",
                              mode="max",
                              patience=5)
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         callbacks=[callback, callback2],
                         # auto_scale_batch_size='binsearch',
                         # auto_lr_find=True,
                         val_check_interval=None,
                         max_epochs=args.max_epochs,
                         logger=logger,
                         log_every_n_steps=50,
                         num_sanity_val_steps=2,
                         fast_dev_run=False,
                         enable_progress_bar=True)
    
    if args.force_restart:
        trainer.fit(model=model,
                    train_dataloaders=DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=0
                                                ),
                    val_dataloaders=DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=0
                                            ),
                    )
        trainer.test(ckpt_path=callback.best_model_path,
                    dataloaders=DataLoader(test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=0
                                            ),
                    )
    else:
        checkpoint = torch.load(args.ckpt)

        model.load_state_dict(checkpoint["state_dict"])

        trainer.test(model=model,
                    dataloaders=DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=0
                                        ),
                    )



def ele_main(args, v_num, train_dataset, val_dataset, test_dataset):
    cat_features = ['user_id', 'gender', 'visit_city', 'is_super_vip', 'city_id', 'district_id', 'shop_geohash_12',
                      'geohash12', 'shop_id_list', 'item_id_list', 'category_1_id_list', 'merge_standard_food_id_list',
                      'brand_id_list', 'shop_aoi_id_list', 'shop_geohash_6_list', 'hours_list', 'time_type_list',
                      'weekdays_list']
    num_features = ['avg_price', 'ctr_30', 'ord_30', 'total_amt_30', 'rank_7', 'rank_30', 'rank_90']
    dnn_col = ['user_id', 'gender', 'visit_city', 'avg_price', 'is_super_vip', 'ctr_30', 'ord_30',
               'total_amt_30', 'city_id', 'district_id', 'shop_geohash_12', 'rank_7', 'rank_30', 'rank_90',
               'geohash12']
    transformer_col = ['shop_id_list', 'item_id_list', 'category_1_id_list', 'merge_standard_food_id_list',
                       'brand_id_list', 'shop_aoi_id_list', 'shop_geohash_6_list', 'hours_list',
                       'time_type_list', 'weekdays_list']
    target_col = 'label'
    time_col = 'timediff_list'
    pl.seed_everything(args.seed)
    model = BST(cat_features=cat_features,
                num_features=num_features,
                dnn_col=dnn_col,
                transformer_col=transformer_col,
                target_col=target_col,
                time_col=time_col,
                args=args)
    logger = TensorBoardLogger(save_dir='logs/ele_logs/train' if args.force_restart else 'logs/ele_logs/test',
                               name='fixed',
                               version=v_num)
    
    callback = ModelCheckpoint(monitor="val/auc",
                mode="max",
                save_top_k=1,
                filename='epoch={epoch}-step={step}-val_auc={val/auc:.4f}-log_loss={val\loss:.4f}',
                save_weights_only=True,
                auto_insert_metric_name=False)
    callback2 = EarlyStopping(monitor="val/auc",
                            mode="max",
                            patience=1)
    trainer = pl.Trainer(accelerator='gpu',
                        devices=1,
                        callbacks=[callback, callback2],
                        # auto_scale_batch_size='binsearch',
                        # auto_lr_find=True,
                        val_check_interval=None,
                        max_epochs=args.max_epochs,
                        logger=logger,
                        log_every_n_steps=50,
                        num_sanity_val_steps=2,
                        fast_dev_run=False,
                        enable_progress_bar=True)
                        
    if args.force_restart:
        trainer.fit(model=model,
                    train_dataloaders=DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=0
                                                ),
                    val_dataloaders=DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=0
                                            ),
                    )

        trainer.test(ckpt_path=callback.best_model_path,
                    dataloaders=DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=0
                                        ),
                    )

    else:
        checkpoint = torch.load(args.ckpt)

        model.load_state_dict(checkpoint["state_dict"])

        trainer.test(model=model,
                    dataloaders=DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=0
                                        ),
                    )


def movie_main(args, v_num, train_dataset, val_dataset, test_dataset):
    dnn_col = ['user_id', 'sex', 'age_group', 'occupation', 'zip_code']
    transformer_col = ['movie_ids', 'years', 'Actions', 'Adventures', 'Animations', 'Childrens',
                               'Comedys', 'Crimes', 'Documentarys', 'Dramas', 'Fantasys', 'Film_Noirs', 'Horrors',
                               'Musicals', 'Mysterys', 'Romances', 'Sci_Fis', 'Thrillers', 'Wars', 'Westerns']

    cat_features = dnn_col + transformer_col
    num_features = []
    target_col = 'label'
    time_col = 'timestamps'
    pl.seed_everything(args.seed)
    model = BST(cat_features=cat_features,
                num_features=num_features,
                dnn_col=dnn_col,
                transformer_col=transformer_col,
                target_col=target_col,
                time_col=time_col,
                args=args)
    logger = TensorBoardLogger(save_dir='logs/movie_logs/train' if args.force_restart else 'logs/movie_logs/test',
                               name='fixed',
                               version=v_num)
    callback = ModelCheckpoint(monitor="val/auc",
                               mode="max",
                               save_top_k=1,
                               filename='epoch={epoch}-step={step}-val_auc={val/auc:.4f}-log_loss={val\loss:.4f}',
                               save_weights_only=True,
                               auto_insert_metric_name=False)
    callback2 = EarlyStopping(monitor="val/auc",
                              mode="max",
                              patience=5)
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         callbacks=[callback, callback2],
                         # auto_scale_batch_size='binsearch',
                         # auto_lr_find=True,
                         val_check_interval=None,
                         max_epochs=args.max_epochs,
                         logger=logger,
                         log_every_n_steps=50,
                         num_sanity_val_steps=2,
                         fast_dev_run=False,
                         enable_progress_bar=True)
    if args.force_restart:
        trainer.fit(model=model,
                    train_dataloaders=DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=0
                                                ),
                    val_dataloaders=DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=0
                                            ),
                    )
        trainer.test(ckpt_path=callback.best_model_path,
                    dataloaders=DataLoader(test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=0
                                            ),
                    )
    else:
        checkpoint = torch.load(args.ckpt)

        model.load_state_dict(checkpoint["state_dict"])

        trainer.test(model=model,
                    dataloaders=DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=0
                                        ),
                    )


def run_bundle_main():
    train_dataset = BundleDataset(os.path.join(args.data_path, "train_data.csv"), args.max_len) if args.force_restart else None
    val_dataset = BundleDataset(os.path.join(args.data_path, "val_data.csv"), args.max_len) if args.force_restart else None
    test_dataset = BundleDataset(os.path.join(args.data_path, "test_data.csv"), args.max_len, True)

    for hparam in hparams:
        args.use_time, args.use_int, args.int_num, args.log_base, args.ckpt = hparam
        print(vars(args))
        bundle_main(args, f"lr={args.lr} mean_std=1 res=concat  use_time={args.use_time} use_int={args.use_int} log_base={args.log_base} int_num={args.int_num}", train_dataset, val_dataset, test_dataset)

def run_ele_main():
    train_dataset = EleDataset(os.path.join(args.data_path, "train_data.csv"), args.max_len) if args.force_restart else None
    val_dataset = EleDataset(os.path.join(args.data_path, "val_data.csv"), args.max_len) if args.force_restart else None
    test_dataset = EleDataset(os.path.join(args.data_path, "test_data.csv"), args.max_len, True)

    for hparam in hparams:
        args.use_time, args.use_int, args.int_num, args.log_base, args.ckpt = hparam
        print(vars(args))
        ele_main(args, f"lr={args.lr} mean_std=1 res=concat  use_time={args.use_time} use_int={args.use_int} log_base={args.log_base} int_num={args.int_num}", train_dataset, val_dataset, test_dataset)


def run_movie_main():
    train_dataset = MovieDataset(os.path.join(args.data_path, "train_data.csv"), args.max_len) if args.force_restart else None
    val_dataset = MovieDataset(os.path.join(args.data_path, "val_data.csv"), args.max_len) if args.force_restart else None
    test_dataset = MovieDataset(os.path.join(args.data_path, "test_data.csv"), args.max_len, True)

    for hparam in hparams:
        args.use_time, args.use_int, args.int_num, args.log_base, args.ckpt = hparam
        print(vars(args))
        movie_main(args, f"lr={args.lr} mean_std=1 res=concat  use_time={args.use_time} use_int={args.use_int} log_base={args.log_base} int_num={args.int_num}", train_dataset, val_dataset, test_dataset)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--use_time', action="store_true")  # True表示使用时间信息，否则使用位置嵌入
    parser.add_argument('--use_int', action="store_true")  # 是否使用特征交互层
    parser.add_argument('--int_num', default=2, type=int)  # 使用特征交互层的层数,-1表示不加残差连接的一层
    parser.add_argument('--log_base', default=10, type=float)  # log_base>1时表示对数时间函数的底数，log_base=-1时表示使用动态底数，log_base=-2时表示使用线性时间函数
    parser.add_argument("--transformer_num", default=2, type=int)  # transformer层的层数
    parser.add_argument("--embedding", default=8, type=int)  # 类别特征的embedding向量维度
    parser.add_argument("--num_head", default=8, type=int)  # transformer层中多头自注意力的头数
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--max_len', default=8, type=int)  # 最长序列长度
    parser.add_argument('--force_restart', action="store_true")  # 是否从头训练模型
    parser.add_argument('--ckpt', type=str)  # 预训练模型路径
    parser.set_defaults(max_epochs=50)
    args = parser.parse_args()

    hparams = [
        # [True, False, 0, 5, None if args.force_restart else find_ckpt_file(f"logs/{args.dataset}_logs/LogTE_base5/checkpoints/")],  # LogTE+base5
        [False, False, 0, 0, None if args.force_restart else find_ckpt_file(f"logs/{args.dataset}_logs/BST/checkpoints/")],  # BST
        # [True, True, 1, 5, None if args.force_restart else find_ckpt_file(f"logs/{args.dataset}_logs/LogTE+FC+base5+numFC1/checkpoints/")],  # LogTE+FC+base5+numFC1
        # [True, True, 1, 2, None if args.force_restart else find_ckpt_file(f"logs/{args.dataset}_logs/LogTE+FC+base2+numFC1/checkpoints/")],  # LogTE+FC+base2+numFC1
        # [True, True, 1, 10, None if args.force_restart else find_ckpt_file(f"logs/{args.dataset}_logs/LogTE+FC+base10+numFC1/checkpoints/")],  # LogTE+FC+base10+numFC1
        # [True, True, 2, 5, None if args.force_restart else find_ckpt_file(f"logs/{args.dataset}_logs/LogTE+FC+base5+numFC2/checkpoints/")],  # LogTE+FC+base5+numFC2
        # [True, True, 3, 5, None if args.force_restart else find_ckpt_file(f"logs/{args.dataset}_logs/LogTE+FC+base5+numFC3/checkpoints/")],  # LogTE+FC+base5+numFC3
        [False, True, 1, 0, None if args.force_restart else find_ckpt_file(f"logs/{args.dataset}_logs/PE+FC+base5+numFC1/checkpoints/")],  # PE+FC+numFC1
        # [True, True, 1, -2, None if args.force_restart else find_ckpt_file(f"logs/{args.dataset}_logs/LinearTE+FC+base5+numFC1/checkpoints/")],  # LinearTE+FC+numFC1
        [False, True, 2, 0, None if args.force_restart else find_ckpt_file(f"logs/{args.dataset}_logs/PE+FC+base5+numFC2/checkpoints/")],  # PE+FC+numFC2
        [False, True, 3, 0, None if args.force_restart else find_ckpt_file(f"logs/{args.dataset}_logs/PE+FC+base5+numFC3/checkpoints/")],  # PE+FC+numFC3
    ]
    if args.dataset == "ele":
        run_ele_main()
    elif args.dataset == "movie":
        run_movie_main()
    elif args.dataset == "bundle":
        run_bundle_main()
