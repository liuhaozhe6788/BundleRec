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
from data.taobao_dataset import TaobaoDataset
from model.BST import BST


def bundle_main(args, v_num, train_dataset, val_dataset, test_dataset):
    spare_features = ['uid', 'is_visitor', 'gender', 'is_up_to_date_version', 'register_country', 'register_device',
                      'register_device_brand', 'register_os', 'bundle_id']
    dense_features = ['register_time', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
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
    model = BST(spare_features=spare_features,
                dense_features=dense_features,
                dnn_col=dnn_col,
                transformer_col=transformer_col,
                target_col=target_col,
                time_col=time_col,
                args=args)
    logger = TensorBoardLogger(save_dir='logs/bundle_logs',
                               name='bundle_BST_final',
                               version=v_num)
    callback = ModelCheckpoint(monitor="val/auc",
                               mode="max",
                               save_top_k=1,
                               filename='epoch={epoch}-step={step}-val_auc={val\auc:.4f}-log_loss={val\loss:.4f}',
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
    spare_features = ['user_id', 'gender', 'visit_city', 'is_super_vip', 'city_id', 'district_id', 'shop_geohash_12',
                      'geohash12', 'shop_id_list', 'item_id_list', 'category_1_id_list', 'merge_standard_food_id_list',
                      'brand_id_list', 'shop_aoi_id_list', 'shop_geohash_6_list', 'hours_list', 'time_type_list',
                      'weekdays_list']
    dense_features = ['avg_price', 'ctr_30', 'ord_30', 'total_amt_30', 'rank_7', 'rank_30', 'rank_90']
    dnn_col = ['user_id', 'gender', 'visit_city', 'avg_price', 'is_super_vip', 'ctr_30', 'ord_30',
               'total_amt_30', 'city_id', 'district_id', 'shop_geohash_12', 'rank_7', 'rank_30', 'rank_90',
               'geohash12']
    transformer_col = ['shop_id_list', 'item_id_list', 'category_1_id_list', 'merge_standard_food_id_list',
                       'brand_id_list', 'shop_aoi_id_list', 'shop_geohash_6_list', 'hours_list',
                       'time_type_list', 'weekdays_list']
    target_col = 'label'
    time_col = 'timediff_list'
    pl.seed_everything(args.seed)
    model = BST(spare_features=spare_features,
                dense_features=dense_features,
                dnn_col=dnn_col,
                transformer_col=transformer_col,
                target_col=target_col,
                time_col=time_col,
                args=args)
    logger = TensorBoardLogger(save_dir='logs\ele_logs\ele_BST',
                               name='fixed',
                               version=v_num)
    
    callback = ModelCheckpoint(monitor="val\auc",
                mode="max",
                save_top_k=1,
                filename='epoch={epoch}-step={step}-val_auc={val\auc:.4f}-log_loss={val\loss:.4f}',
                save_weights_only=True,
                auto_insert_metric_name=False)
    callback2 = EarlyStopping(monitor="val\auc",
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

    spare_features = dnn_col + transformer_col
    dense_features = []
    target_col = 'label'
    time_col = 'timestamps'
    pl.seed_everything(args.seed)
    model = BST(spare_features=spare_features,
                dense_features=dense_features,
                dnn_col=dnn_col,
                transformer_col=transformer_col,
                target_col=target_col,
                time_col=time_col,
                args=args)
    logger = TensorBoardLogger(save_dir='.\logs\movie_logs',
                               name='movie_BST',
                               version=v_num)
    callback = ModelCheckpoint(monitor="val/auc",
                               mode="max",
                               save_top_k=1,
                               filename='epoch={epoch}-step={step}-val_auc={val\auc:.4f}-log_loss={val\loss:.4f}',
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

def taobao_main(args, v_num, train_dataset, val_dataset, test_dataset):
    dnn_col = ['user_id']
    transformer_col = ['product_ids', 'product_category_ids']

    spare_features = dnn_col + transformer_col
    dense_features = []
    target_col = 'label'
    time_col = 'timestamps'
    pl.seed_everything(args.seed)
    model = BST(spare_features=spare_features,
                dense_features=dense_features,
                dnn_col=dnn_col,
                transformer_col=transformer_col,
                target_col=target_col,
                time_col=time_col,
                args=args)
    logger = TensorBoardLogger(save_dir='.\logs\taobao_logs',
                               name='taobao_BST',
                               version=v_num)
    callback = ModelCheckpoint(monitor="val\auc",
                               mode="max",
                               save_top_k=1,
                               filename='epoch={epoch}-step={step}-val_auc={val\auc:.4f}-log_loss={val\loss:.4f}',
                               save_weights_only=True,
                               auto_insert_metric_name=False)
    callback2 = EarlyStopping(monitor="val\auc",
                              mode="max",
                              patience=3)
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

    v_num = None
    # for seed in [0]:
    #     for use_time, log_base in [(True, 10)]:
    #         for use_int, int_num, embedding in [(True, 1, 8), (True, 2, 8), (True, 3, 8)]:
    #             args.seed, args.log_base, args.int_num, args.use_time, args.use_int, args.embedding \
    #                 = seed, log_base, int_num, use_time, use_int, embedding
    #             bundle_main(args, v_num, train_dataset, val_dataset, test_dataset)

    if args.force_restart:
        for seed in [0]:
            for use_time, log_base in [(True, 10)]:
                for use_int, int_num, embedding in [(True, -1, 8)]:
                    args.seed, args.log_base, args.int_num, args.use_time, args.use_int, args.embedding \
                        = seed, log_base, int_num, use_time, use_int, embedding
                    bundle_main(args, f"train lr={args.lr} mean_std=1 res=concat  use_time={args.use_time} use_int={args.use_int} log_base={args.log_base} int_num={args.int_num}", train_dataset, val_dataset, test_dataset)

        for seed in [0]:
            for use_time, log_base in [(True, -2), (True, -3)]:
                for use_int, int_num, embedding in [(False, 0, 9)]:
                    args.seed, args.log_base, args.int_num, args.use_time, args.use_int, args.embedding \
                        = seed, log_base, int_num, use_time, use_int, embedding
                    bundle_main(args, f"train lr={args.lr} mean_std=1 res=concat  use_time={args.use_time} use_int={args.use_int} log_base={args.log_base} int_num={args.int_num}", train_dataset, val_dataset, test_dataset)
    else:
        hparams = [
            [False, True, 1, 0, 8, 2001, "logs/bundle_logs/version_0/checkpoints/epoch=7-step=52424-val_auc=0.8672-log_loss=0.0155.ckpt"],
            [False, False, 0, 0, 9, 2001, "logs/bundle_logs/version_1/checkpoints/epoch=6-step=45871-val_auc=0.8519-log_loss=0.0148.ckpt"],
            [True, True, 1, 10, 8, 51, "logs/bundle_logs/version_2/checkpoints/epoch=8-step=58977-val_auc=0.8648-log_loss=0.0142.ckpt"],
            [True, False, 0, 2, 9, 2001, "logs/bundle_logs/version_22/checkpoints/epoch=9-step=65530-val_auc=0.8516-log_loss=0.0144.ckpt"],
            [True, False, 0, 5, 9, 2001, "logs/bundle_logs/version_23/checkpoints/epoch=7-step=52424-val_auc=0.8578-log_loss=0.0142.ckpt"],
            [True, False, 0, -2, 9, 2001, "logs/bundle_logs/version_29/checkpoints/epoch=9-step=65530-val_auc=0.8380-log_loss=0.0148.ckpt"],
            [True, False, 0, 10, 9, 51, "logs/bundle_logs/version_3/checkpoints/epoch=7-step=52424-val_auc=0.8585-log_loss=0.0149.ckpt"],
        ]
        for hparam in hparams:
            args.use_time, args.use_int, args.int_num, args.log_base, args.embedding, args.num_embeds, args.ckpt = hparam
            print(vars(args))
            bundle_main(args, f"lr={args.lr} mean_std=1 res=concat  use_time={args.use_time} use_int={args.use_int} log_base={args.log_base} int_num={args.int_num}", train_dataset, val_dataset, test_dataset)


def run_ele_main():
    train_dataset = EleDataset(os.path.join(args.data_path, "train_data.csv"), args.max_len) if args.force_restart else None
    val_dataset = EleDataset(os.path.join(args.data_path, "val_data.csv"), args.max_len) if args.force_restart else None
    test_dataset = EleDataset(os.path.join(args.data_path, "test_data.csv"), args.max_len, True)

    if args.force_restart:
        for seed in [0]:
            for transformer_num in [1]:
                for use_time, log_base in [(True, 5), (False, 0)]:
                    for use_int, int_num in [(True, 1), (True, 2), (True, 3)]:
                        for lr in [1e-5]:
                            args.seed, args.transformer_num, args.log_base, args.int_num, args.use_time, args.use_int, args.lr \
                                    = seed, transformer_num, log_base, int_num, use_time, use_int, lr
                            ele_main(args, f"train lr={args.lr} mean_std=1 res=concat use_time={args.use_time} use_int={args.use_int} log_base={log_base} int_num={int_num}", train_dataset, val_dataset, test_dataset)
    else:
        hparams = [
            [1e-5, True, False, 0, 5, 50, "logs/ele_logs/version_108/checkpoints/epoch=4-step=15565-val_auc=0.6267-log_loss=0.0886.ckpt"],
            [1e-5, True, False, 0, -2, 2001, "logs/ele_logs/version_109/checkpoints/epoch=2-step=9339-val_auc=0.6245-log_loss=0.0889.ckpt"],
            [1e-5, True, False, 0, 2, 51, "logs/ele_logs/version_28/checkpoints/epoch=4-step=15565-val_auc=0.6252-log_loss=0.0887.ckpt"],
            [1e-5, True, False, 0, 10, 51, "logs/ele_logs/version_32/checkpoints/epoch=4-step=15565-val_auc=0.6271-log_loss=0.0886.ckpt"],
            [1e-3, False, True, 3, 0, 2001, "logs/ele_logs/version_124/checkpoints/epoch=1-step=6226-val_auc=0.5943-log_loss=0.0898.ckpt"],
            [1e-4, False, True, -1, 0, 2001, "logs/ele_logs/version_126/checkpoints/epoch=3-step=12452-val_auc=0.5871-log_loss=0.0896.ckpt"],
            [1e-5, False, False, 0, 0, 2001, "logs/ele_logs/version_24/checkpoints/epoch=6-step=21791-val_auc=0.5963-log_loss=0.0895.ckpt"],
            [1e-5, False, True, 1, 0, 2001, "logs/ele_logs/version_25/checkpoints/epoch=20-step=65373-val_auc=0.5985-log_loss=0.0894.ckpt"],
            [1e-5, False, True, 2, 0, 2001, "logs/ele_logs/version_26/checkpoints/epoch=27-step=87164-val_auc=0.5924-log_loss=0.0896.ckpt"]
        ]
        for hparam in hparams:
            args.lr, args.use_time, args.use_int, args.int_num, args.log_base, args.num_embeds, args.ckpt = hparam
            print(vars(args))
            ele_main(args, f"lr={args.lr} mean_std=1 res=concat  use_time={args.use_time} use_int={args.use_int} log_base={args.log_base} int_num={args.int_num}", train_dataset, val_dataset, test_dataset)


def run_movie_main():
    train_dataset = MovieDataset(os.path.join(args.data_path, "train_data.csv"), args.max_len) if args.force_restart else None
    val_dataset = MovieDataset(os.path.join(args.data_path, "val_data.csv"), args.max_len) if args.force_restart else None
    test_dataset = MovieDataset(os.path.join(args.data_path, "test_data.csv"), args.max_len, True)

    hparams = [
        [False, 0, 2001, "logs/movie_logs/version_12/checkpoints/epoch=10-step=12078-val_auc=0.7855-log_loss=0.4433.ckpt"],
        [True, 1, 2001, "logs/movie_logs/version_14/checkpoints/epoch=21-step=24156-val_auc=0.7840-log_loss=0.4427.ckpt"],
        [True, 2, 2001, "logs/movie_logs/version_53/checkpoints/epoch=29-step=32940-val_auc=0.7849-log_loss=0.4502.ckpt"],
        [True, 3, 2001, "logs/movie_logs/version_54/checkpoints/epoch=27-step=30744-val_auc=0.7755-log_loss=0.4632.ckpt"],
        [True, -1, 2001, "logs/movie_logs/version_56/checkpoints/epoch=34-step=38430-val_auc=0.0000-log_loss=0.0000.ckpt"]
        ]

    for hparam in hparams:
        if args.force_restart:
            args.use_int, args.int_num, args.num_embeds, _ = hparam
        else:
            args.use_int, args.int_num, args.num_embeds, args.ckpt = hparam
        print(vars(args))
        movie_main(args, f"lr={args.lr} mean_std=1 res=concat  use_time={args.use_time} use_int={args.use_int} log_base={args.log_base} int_num={args.int_num}", train_dataset, val_dataset, test_dataset)


def run_taobao_main():

    train_dataset = TaobaoDataset(os.path.join(args.data_path, "train_data.csv"), args.max_len) if args.force_restart else None
    val_dataset = TaobaoDataset(os.path.join(args.data_path, "val_data.csv"), args.max_len) if args.force_restart else None
    test_dataset = TaobaoDataset(os.path.join(args.data_path, "test_data.csv"), args.max_len, True)

    if args.force_restart:
        for seed in [0]:
            for use_time, log_base in [(True, -2)]:
                for use_int, int_num in [(False, 0)]:
                    args.seed, args.log_base, args.use_time, args.use_int, args.int_num \
                        = seed, log_base, use_time, use_int, int_num
                    taobao_main(args, f"train lr={args.lr} mean_std=1 res=concat use_time={args.use_time} use_int={args.use_int} log_base={log_base} int_num={int_num}", train_dataset, val_dataset, test_dataset)
    else:
        hparams = [
            [True, False, 0, 10, 51, "logs/taobao_logs/version_2/checkpoints/epoch=8-step=263061-val_auc=0.7488-log_loss=0.0950.ckpt"],
            [True, False, 0, 5, 2001, "logs/taobao_logs/version_26/checkpoints/epoch=7-step=233832-val_auc=0.7583-log_loss=0.0932.ckpt"],
            [True, False, 0, 2, 2001, "logs/taobao_logs/version_27/checkpoints/epoch=6-step=204603-val_auc=0.7609-log_loss=0.0907.ckpt"],
            [True, False, 0, -2, 2001, "logs/taobao_logs/version_30/checkpoints/epoch=7-step=233832-val_auc=0.7523-log_loss=0.0930.ckpt"],
            [False, False, 0, 0, 2001, "logs/taobao_logs/version_4/checkpoints/epoch=4-step=146145-val_auc=0.7293-log_loss=0.0939.ckpt"]
        ]
        for hparam in hparams:
            args.use_time, args.use_int, args.int_num, args.log_base, args.num_embeds, args.ckpt = hparam
            print(vars(args))
            taobao_main(args, f"lr={args.lr} mean_std=1 res=concat  use_time={args.use_time} use_int={args.use_int} log_base={args.log_base} int_num={args.int_num}", train_dataset, val_dataset, test_dataset)


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
    parser.add_argument('--num_embeds', default=51, type=int)  # 嵌入层的元素个数
    parser.add_argument('--force_restart', action="store_true")  # 是否从头训练模型
    parser.add_argument('--ckpt', type=str)  # 预训练模型路径
    parser.set_defaults(max_epochs=50)
    args = parser.parse_args()
    if args.dataset == "ele":
        run_ele_main()
    elif args.dataset == "taobao":
        run_taobao_main()
    elif args.dataset == "movie":
        run_movie_main()
    elif args.dataset == "bundle":
        run_bundle_main()
