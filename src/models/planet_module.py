from typing import Any, List, Tuple, Dict

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from lightning import LightningModule

from src.models.hub.numformer import Numformer
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

class PlanetModule(LightningModule):
    def __init__(
        self,
        net: Numformer,
        tokenizer_path,
        lr: float = 2e-5,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_iterations: int = 2000,
        max_iterations: int = 500000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 2e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['net'])
        
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token="[END]",
            eos_token="[END]",
            mask_token="[MASK]",
            pad_token="[PAD]",
        )
        self.tokenizer = tokenizer
        self.num_token_id = tokenizer.convert_tokens_to_ids("[NUM]")
        
        self.net = net
        
    def training_step(self, batch: Any, batch_idx: int):
        logit_preds, num_preds = self.net(batch["x"], batch["x_num"])
        loss_mlm = F.cross_entropy(
            logit_preds.view(-1, logit_preds.size(-1)),
            batch["y"].view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        
        num_mask = batch['y'] == self.num_token_id
        loss_num = F.mse_loss(
            num_preds[num_mask],
            batch["y_num"][num_mask].view(-1,1),
            reduction="mean",
        )
        
        loss_dict = {"loss_mlm": loss_mlm, "loss_num": loss_num, "loss": loss_mlm + loss_num}
        for key, value in loss_dict.items():
            self.log(
                f"train/{key}",
                value.item(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=batch["x"].size(0),
            )
        
        return loss_dict["loss"]
    
    def validation_step(self, batch: Any, batch_idx: int):
        logit_preds, num_preds = self.net(batch["x"], batch["x_num"])
        loss_mlm = F.cross_entropy(
            logit_preds.view(-1, logit_preds.size(-1)),
            batch["y"].view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        
        num_mask = batch['y'] == self.num_token_id
        loss_num = F.mse_loss(
            num_preds[num_mask],
            batch["y_num"][num_mask].view(-1,1),
            reduction="mean",
        )
        
        loss_dict = {"loss_mlm": loss_mlm, "loss_num": loss_num, "loss": loss_mlm + loss_num}
        for key, value in loss_dict.items():
            self.log(
                f"val/{key}",
                value.item(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=batch["x"].size(0),
            )
        
        return loss_dict
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta_1, self.hparams.beta_2),
            weight_decay=self.hparams.weight_decay,
        )
        
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_iterations,
            max_epochs=self.hparams.max_iterations,
            warmup_start_lr=self.hparams.warmup_start_lr,
            eta_min=self.hparams.eta_min,
        )
        lr_scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}