import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)

from datasets.classification_dataset import (BUSIImageDataset,
                                            AUIDTImageDataset)
from datasets.data_module import DataModule
from datasets.transforms import DataTransforms, Moco2Transform
from models.QFT.QFT_training import QFT
from models.ssl_finetuner import SSLFineTuner

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chexpert")
    parser.add_argument("--path", type=str,
                        default="/home/sutongkun/VLPv2/QFT/data/ckpts/QFT/18/epoch=2-step=260.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--data_pct", type=float, default=1)

    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set max epochs
    args.max_epochs = 50

    seed_everything(args.seed)

    if args.dataset == "busi":
        datamodule = DataModule(BUSIImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 3
        multilabel = False
    elif args.dataset == "auidt":
        datamodule = DataModule(AUIDTImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 3
        multilabel = False
    else:
        raise RuntimeError(f"no dataset called {args.dataset}")

    if args.path:
         model = QFT.load_from_checkpoint(args.path, strict=False)
    else:
        model = QFT()

    args.model_name = model.hparams.img_encoder
    args.backbone = model.img_encoder_q
    # sd = torch.load('vit.pth')
    # from collections import OrderedDict
    # new_sd = OrderedDict()
    # for k, v in sd.items():
    #     new_sd['model.'+k] = v
    args.backbone.load_state_dict(torch.load('/home/sutongkun/VLPv2/GLoRIA/data/ckpt/gloria_pretrain_1.0/2023_10_17_10_04_50/new.pth'), strict=False)
    args.in_features = args.backbone.feature_dim
    args.num_classes = num_classes
    args.multilabel = multilabel

    # finetune
    tuner = SSLFineTuner(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../data/ckpts/QFT_CLS/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]

    # get current time
    now = datetime.datetime.now(tz.tzlocal())

    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    logger_dir = os.path.join(
        BASE_DIR, f"../../data")
    os.makedirs(logger_dir, exist_ok=True)
    trainer = Trainer.from_argparse_args(
        args,
        deterministic=True,
        callbacks=callbacks)

    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)

    # train
    trainer.fit(tuner, datamodule)
    # test
    trainer.test(tuner, datamodule, ckpt_path="best")

if __name__ == "__main__":
    cli_main()
