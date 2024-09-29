import datetime
import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from datasets.data_module import DataModule
from datasets.pretrain_dataset import MultimodalPretrainingDataset, multimodal_collate_fn
from datasets.transforms import DataTransforms
from models.backbones.encoder import BertEncoder, ImageEncoder
from transformers import GPT2LMHeadModel, GPT2Config

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class QFT(LightningModule):
    '''Pytorch lightning implementation of QFT'''

    def __init__(self,
                 img_encoder: str = "resnet_50",
                 freeze_bert: bool = False,
                 queue_size = 100,
                 query_num = 32,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 lambda_1: float = 1,
                 lambda_2: float = 0.7,
                 lambda_3: float = 0.5,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert, split='text_encoder')
        self.qft = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert, split='qft')
        config = GPT2Config.from_json_file(os.path.join(BASE_DIR, "../../configs/gpt_config.json"))
        self.decoder = GPT2LMHeadModel.from_pretrained(os.path.join(BASE_DIR, "../backbones/GPT"),
                                                       config=config)

        self.register_buffer("img_queue", torch.randn(self.hparams.queue_size, self.hparams.emb_dim))
        self.register_buffer("text0_queue", torch.randn(self.hparams.queue_size, self.hparams.emb_dim))
        self.register_buffer("text1_queue", torch.randn(self.hparams.queue_size, self.hparams.emb_dim))
        self.register_buffer("qfeat0_queue", torch.randn(self.hparams.queue_size, self.hparams.query_num, self.hparams.emb_dim))
        self.register_buffer("qfeat1_queue", torch.randn(self.hparams.queue_size, self.hparams.query_num, self.hparams.emb_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.img_queue = F.normalize(self.img_queue, dim=1)
        self.text0_queue = F.normalize(self.text0_queue, dim=1)
        self.text1_queue = F.normalize(self.text1_queue, dim=1)
        self.qfeat0_queue = F.normalize(self.qfeat0_queue, dim=2)
        self.qfeat1_queue = F.normalize(self.qfeat1_queue, dim=2)

    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''

        global_img0_feat, local_img0_feat = self.img_encoder_q(batch['img0'])
        global_img1_feat, local_img1_feat = self.img_encoder_q(batch['img1'])
        global_text_feat, local_text_feat = self.text_encoder_q(ids=batch['report'], attn_mask=batch['attn'])

        #### Contrastive Learning ####

        cl_img0_feat = self.img_encoder_q.embed_0(global_img0_feat)
        cl_img0_feat = F.normalize(cl_img0_feat, dim=-1)
        cl_img1_feat = self.img_encoder_q.embed_0(global_img1_feat)
        cl_img1_feat = F.normalize(cl_img1_feat, dim=-1)
        cl_img_feat = (cl_img0_feat + cl_img1_feat) / 2
        cl_text_feat = self.text_encoder_q.embed_0(global_text_feat)
        cl_text_feat = F.normalize(cl_text_feat, dim=-1)

        cl_img_feat_all = torch.cat([cl_img_feat, self.img_queue.clone().detach()], dim=0)
        cl_text_feat_all = torch.cat([cl_text_feat, self.text0_queue.clone().detach()], dim=0)

        bs = cl_text_feat.size(0)
        cl_labels = torch.arange(bs).type_as(cl_text_feat).long()

        scores = cl_img_feat.mm(cl_text_feat_all.t()) / self.hparams.softmax_temperature
        scores_t = cl_text_feat.mm(cl_img_feat_all.t()) / self.hparams.softmax_temperature
        closs0 = F.cross_entropy(scores, cl_labels)
        closs1 = F.cross_entropy(scores_t, cl_labels)
        c_loss = closs0 + closs1

        i2t_acc1, i2t_acc5 = self.precision_at_k(
            scores, cl_labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            scores_t, cl_labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        #### Q Contrastive Learning ####

        lm_img0_feat = self.img_encoder_q.embed_1(local_img0_feat)
        lm_img0_feat = F.normalize(lm_img0_feat, dim=-1)
        lm_img1_feat = self.img_encoder_q.embed_1(local_img1_feat)
        lm_img1_feat = F.normalize(lm_img1_feat, dim=-1)

        qft_input = torch.zeros([bs, self.hparams.query_num, self.hparams.hidden_dim]).to(lm_img0_feat.device)
        global_q_feat0, local_q_feat0 = self.qft(inputs_embeds=qft_input, encoder_hidden_states=lm_img0_feat)
        q_feat0 = torch.cat([global_q_feat0.unsqueeze(1), local_q_feat0], dim=1)
        global_q_feat1, local_q_feat1 = self.qft(inputs_embeds=qft_input, encoder_hidden_states=lm_img1_feat)
        q_feat1 = torch.cat([global_q_feat1.unsqueeze(1), local_q_feat1], dim=1)
        qcl_q_feat0 = self.qft.embed_0(q_feat0)
        qcl_q_feat0 = F.normalize(qcl_q_feat0, dim=-1)
        qcl_q_feat1 = self.qft.embed_0(q_feat1)
        qcl_q_feat1 = F.normalize(qcl_q_feat1, dim=-1)

        qcl_text_feat = self.text_encoder_q.embed_1(global_text_feat)
        qcl_text_feat = F.normalize(qcl_text_feat, dim=-1)

        qcl_q_feat0_all = torch.cat([qcl_q_feat0, self.qfeat0_queue.clone().detach()], dim=0)
        qcl_q_feat1_all = torch.cat([qcl_q_feat1, self.qfeat1_queue.clone().detach()], dim=0)
        qcl_text_feat_all = torch.cat([qcl_text_feat, self.text1_queue.clone().detach()], dim=0)

        sim_mat0 = torch.matmul(qcl_text_feat,
                                torch.transpose(qcl_q_feat0_all, 1, 2)) / self.hparams.softmax_temperature
        qscores0 = torch.max(sim_mat0, dim=-1)[0]
        qscores0 = qscores0.t()
        sim_mat1 = torch.matmul(qcl_text_feat,
                                torch.transpose(qcl_q_feat1_all, 1, 2)) / self.hparams.softmax_temperature
        qscores1 = torch.max(sim_mat1, dim=-1)[0]
        qscores1 = qscores1.t()
        qscores = (qscores0 + qscores1)/2

        sim_mat2 = torch.matmul(qcl_q_feat0,
                                qcl_text_feat_all.t()) / self.hparams.softmax_temperature
        qscores2 = torch.max(sim_mat2, dim=1)[0]
        sim_mat3 = torch.matmul(qcl_q_feat1,
                                qcl_text_feat_all.t()) / self.hparams.softmax_temperature
        qscores3 = torch.max(sim_mat3, dim=1)[0]
        qscores_t = (qscores2 + qscores3) / 2

        qloss0 = F.cross_entropy(qscores, cl_labels)
        qloss1 = F.cross_entropy(qscores_t, cl_labels)
        q_loss = qloss0 + qloss1

        self._dequeue_and_enqueue(cl_img_feat, cl_text_feat, qcl_text_feat, qcl_q_feat0.contiguous(), qcl_q_feat1.contiguous())

        #### Language Modeling ####

        q_feat = torch.cat([q_feat0, q_feat1], dim=1)
        q_feat = self.qft.embed_1(q_feat)
        q_feat = F.normalize(q_feat, dim=-1)

        output = self.decoder(input_ids=batch['text0'], attention_mask=batch['attn0'],
                              encoder_hidden_states=q_feat.contiguous(), labels=batch['label0'])
        t_loss0 = output['loss']

        if batch['text1'] != None:
            output = self.decoder(input_ids=batch['text1'], attention_mask=batch['attn1'],
                               encoder_hidden_states=q_feat.contiguous(), labels=batch['label1'])
            t_loss1 = output['loss']
        else:
            t_loss1 = torch.tensor([0]).type_as(t_loss0)

        if batch['text2'] != None:
            output = self.decoder(input_ids=batch['text2'], attention_mask=batch['attn2'],
                               encoder_hidden_states=q_feat.contiguous(), labels=batch['label2'])
            t_loss2 = output['loss']

        else:
            t_loss2 = torch.tensor([0]).type_as(t_loss0)

        lm_loss = self.hparams.lambda_0 * t_loss0 + self.hparams.lambda_1 * t_loss1 + self.hparams.lambda_2 * t_loss2

        return c_loss, q_loss, lm_loss, acc1, acc5

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text0_feat, text1_feat, qfeat0_feat, qfeat1_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text0_feats = concat_all_gather(text0_feat)
        text1_feats = concat_all_gather(text1_feat)
        qfeat0_feats = concat_all_gather(qfeat0_feat)
        qfeat1_feats = concat_all_gather(qfeat1_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.queue_size % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.img_queue[ptr:ptr + batch_size, :] = image_feats
        self.text0_queue[ptr:ptr + batch_size, :] = text0_feats
        self.text1_queue[ptr:ptr + batch_size, :] = text1_feats
        self.qfeat0_queue[ptr:ptr + batch_size, :, :] = qfeat0_feats
        self.qfeat1_queue[ptr:ptr + batch_size, :, :] = qfeat1_feats
        ptr = (ptr + batch_size) % self.hparams.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def encode(self, img0, img1):

        global_img0_feat, local_img0_feat = self.img_encoder_q(img0)
        global_img1_feat, local_img1_feat = self.img_encoder_q(img0)
        bs = len(global_img0_feat)

        lm_img0_feat = torch.cat([global_img0_feat.unsqueeze(1), local_img0_feat], dim=1)
        lm_img0_feat = self.img_encoder_q.embed_1(lm_img0_feat)
        lm_img0_feat = F.normalize(lm_img0_feat, dim=-1)
        lm_img1_feat = torch.cat([global_img1_feat.unsqueeze(1), local_img1_feat], dim=1)
        lm_img1_feat = self.img_encoder_q.embed_1(lm_img1_feat)
        lm_img1_feat = F.normalize(lm_img1_feat, dim=-1)

        qft_input = torch.zeros([bs, 32, 768]).to(lm_img0_feat.device)
        global_q_feat0, local_q_feat0 = self.qft(inputs_embeds=qft_input, encoder_hidden_states=lm_img0_feat)
        q_feat0 = torch.cat([global_q_feat0.unsqueeze(1), local_q_feat0], dim=1)
        global_q_feat1, local_q_feat1 = self.qftg(inputs_embeds=qft_input, encoder_hidden_states=lm_img1_feat)
        q_feat1 = torch.cat([global_q_feat1.unsqueeze(1), local_q_feat1], dim=1)

        q_feat = torch.cat([q_feat0, q_feat1], dim=1)
        q_feat = self.qft.embed_1(q_feat)
        q_feat = F.normalize(q_feat, dim=-1)

        return q_feat

    def decode(self, input_ids, encoder_output):

        output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_output)

        return output


    def training_step(self, batch, batch_idx):
        c_loss, q_loss, lm_loss, acc1, acc5 = self(
            batch, batch_idx, "train")
        loss = lm_loss + q_loss + c_loss

        log = {
            "train_loss": loss,
            "train_contrastive_loss": c_loss,
            "train_q_contrastive_loss": q_loss,
            "train_language_model_loss": lm_loss,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        c_loss, q_loss, lm_loss, acc1, acc5 = self(
            batch, batch_idx, "valid")

        loss = lm_loss + q_loss + c_loss

        log = {
            "val_loss": loss,
            "val_contrastive_loss": c_loss,
            "val_q_contrastive_loss": q_loss,
            "val_language_model_loss": lm_loss,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,  # 重启系数
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,  # 最小学习率
            warmup_steps=int(self.training_steps * 0.4)
        )
        pass
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=25)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_0", type=float, default=9)
        parser.add_argument("--lambda_1", type=float, default=3)
        parser.add_argument("--lambda_2", type=float, default=1)
        # parser.add_argument("--lambda_3", type=float, default=1.)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--data_pct", type=float, default=1.)
        parser.add_argument("--hidden_dim", type=int, default=768)
        parser.add_argument("--query_num", type=int, default=32)
        parser.add_argument("--queue_size", type=int, default=100)
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs

@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def cli_main():
    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = QFT.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 50

    # seed
    seed_everything(args.seed)

    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers)

    # Add load from checkpoint
    model = QFT(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../data/ckpts/QFT/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=2),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=5, verbose=False, mode="min")
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

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
