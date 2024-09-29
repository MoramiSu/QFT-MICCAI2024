import os

import numpy as np
import torch
import torch.nn as nn
from utils.segmentation_loss import MixedLoss
from pytorch_lightning import LightningModule
import cv2
from constants import *
from matplotlib import pyplot as plt

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SSLSegmenter(LightningModule):
    def __init__(self,
                 seg_model: nn.Module,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-6,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['seg_model'])
        self.model = seg_model
        self.loss = MixedLoss(alpha=10)

        # for param in self.model.encoder.parameters():
        #     param.requires_grad = False

    def shared_step(self, batch, batch_idx, split):
        x, y, filename = batch['imgs'], batch['labels'], batch['filenames']
        logit = self.model(x)
        logit = logit.squeeze(dim=1)
        loss = self.loss(logit, y)
        prob = torch.sigmoid(logit)
        dice = self.get_dice(prob, y)

        if batch_idx == 0:
            img = batch['imgs'][0].cpu().numpy()
            mask = batch['labels'][0].cpu().numpy()
            mask = np.stack([mask, mask, mask])

            layered = 0.6 * mask + 0.4 * img
            img = img.transpose((1, 2, 0))
            mask = mask.transpose((1, 2, 0))
            layered = layered.transpose((1, 2, 0))

        # gt = y[0].cpu().numpy()
        # _, gt = cv2.threshold(gt, 0.5, 255, 0)
        # gt = cv2.convertScaleAbs(gt)
        #
        # # Load the image
        # image_path = str(BUSI_IMG_DIR / batch['filenames'][0])
        # image = cv2.imread(image_path)
        # image = cv2.resize(image, (224, 224))
        #
        # # Draw bounding box on the image
        # coutours_gt, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(image, coutours_gt, -1, (0, 0, 255), 2)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.show()

        if split == 'test':
            prob = prob.cpu()
            pred_mask = (prob > 0.5).float()
            for i in range(len(batch['filenames'])):
                # filename = batch['filenames'][i]
                # path = BUSI_SEG_VISUAL_RES_QFT / ('pred_' + filename)
                # cv2.imwrite(str(path), pred_mask[i].numpy()*255)
                gt = y[i].cpu().numpy()
                _, gt = cv2.threshold(gt, 0.5, 255, 0)
                gt = cv2.convertScaleAbs(gt)
                pred = pred_mask[i].cpu().numpy()
                _, pred = cv2.threshold(pred, 0.5, 255, 0)
                pred = cv2.convertScaleAbs(pred)

                image_path = str(BUSI_IMG_DIR / batch['filenames'][i])
                image = cv2.imread(image_path)
                image = cv2.resize(image, (224, 224))

                coutours_gt, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                coutours_pred, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, coutours_gt, -1, (0, 255, 0), 2)
                cv2.drawContours(image, coutours_pred, -1, (0, 0, 255), 2)

                gt_output_path = str(BUSI_SEG_VISUAL_RES_QFT / batch['filenames'][i])
                cv2.imwrite(gt_output_path, image)

        self.log(
            f"{split}_loss",
            loss.item(),
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        return_dict = {"loss": loss, "dice": dice}
        return return_dict

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")

    def shared_epoch_end(self, step_outputs, split):
        loss = [x["loss"].item() for x in step_outputs]
        dice = [x["dice"] for x in step_outputs]
        loss = np.array(loss).mean()
        dice = np.array(dice).mean()

        self.log(f"{split}_dice", dice, on_epoch=True,
                 logger=True, prog_bar=True)

    def training_epoch_end(self, training_step_outputs):
        return self.shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self.shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self.shared_epoch_end(test_step_outputs, "test")

    def get_dice(self, probability, truth, threshold=0.5):
        batch_size = len(truth)
        with torch.no_grad():
            probability = probability.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            assert probability.shape == truth.shape

            p = (probability > threshold).float()
            t = (truth > 0.5).float()

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = torch.nonzero(t_sum == 0)
            pos_index = torch.nonzero(t_sum >= 1)

            dice_neg = (p_sum == 0).float()
            dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

            dice_neg = dice_neg[neg_index]
            dice_pos = dice_pos[pos_index]
            dice = torch.cat([dice_pos, dice_neg])

        return torch.mean(dice).detach().item()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay
        )

        return optimizer

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
