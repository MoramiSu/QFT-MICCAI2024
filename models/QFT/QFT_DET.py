import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)

from datasets.data_module import DataModule
from datasets.detection_dataset import (BUSIDetectionDataset,
                                        DDTIDetectionDataset)
from datasets.transforms import DetectionDataTransforms
from models.backbones.detector_backbone import ResNetDetector
from models.ssl_detector import SSLDetector

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def cli_main():
    parser = ArgumentParser("Finetuning of object detection task for QFT")
    parser.add_argument("--base_model", type=str, default="resnet_50")
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/sutongkun/VLPv2/QFT/data/ckpts/QFT/18/epoch=2-step=260.ckpt")
    parser.add_argument("--dataset", type=str,
                        default="rsna", help="rsna or object_cxr or busi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--data_pct", type=float, default=1)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 50
    args.accelerator = "gpu"

    seed_everything(args.seed)

    if args.dataset == "busi":
        datamodule = DataModule(BUSIDetectionDataset, None, DetectionDataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
    elif args.dataset == "ddti":
        datamodule = DataModule(DDTIDetectionDataset, None, DetectionDataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
    else:
        raise RuntimeError(f"{args.dataset} does not exist!")

    args.img_encoder = ResNetDetector("resnet_50")

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)
        ckpt_dict = dict()
        for k, v in ckpt["state_dict"].items():
            if k.startswith("gloria.img_encoder.model"):
                new_k = ".".join(k.split(".")[3:])
                new_k = new_k.replace("blocks", "layer")
                ckpt_dict[new_k] = v

        msg = args.img_encoder.model.load_state_dict(ckpt_dict)
        print(msg)

    # Freeze encoder
    for param in args.img_encoder.parameters():
        param.requires_grad = False

    model = SSLDetector(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../data/ckpts/detection/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_mAP", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=1),
        EarlyStopping(monitor="val_mAP", min_delta=0.,
                      patience=10, verbose=False, mode="max")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../data")
    os.makedirs(logger_dir, exist_ok=True)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
    )
    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
