from typing import Any, Union
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from lightning import LightningModule

from src.models.hub.numformer import Numformer
from src.models.hub.numformer_mlp import NumformerMLP
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

class MultiModule(LightningModule):
    def __init__(
        self,
        net: Union[Numformer, NumformerMLP],
        tokenizer_path,
        lr: float = 2e-5,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 2000,
        max_epochs: int = 500000,
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
        self.net.set_num_id(self.num_token_id)
        
    def training_step(self, batch: Any, batch_idx: int):
        _, num_preds = self.net(batch["x"], batch["x_num"])
        
        num_mask = batch['y'] == self.num_token_id # ids of masked tokens that are numbers
        loss = F.mse_loss(
            num_preds[num_mask],
            batch["y_num"][num_mask].view(-1,1),
            reduction="mean",
        )
            
        self.log(
            f"train/loss",
            loss.item(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch["x"].size(0),
        )
        
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        _, num_preds = self.net(batch["x"], batch["x_num"])
        dataset_name = batch["dataset_name"]
        
        num_mask = batch['y'] == self.num_token_id # ids of masked tokens that are numbers
        loss = F.mse_loss(
            num_preds[num_mask],
            batch["y_num"][num_mask].view(-1,1),
            reduction="mean",
        )
            
        self.log(
            f"val/loss_{dataset_name}",
            loss.item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["x"].size(0),
            add_dataloader_idx=False,
        )
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta_1, self.hparams.beta_2),
            weight_decay=self.hparams.weight_decay,
        )
        
        n_steps_per_machine = len(self.trainer.datamodule.train_dataloader())
        n_steps = int(n_steps_per_machine / (self.trainer.num_devices * self.trainer.num_nodes))
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs * n_steps,
            self.hparams.max_epochs * n_steps,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        lr_scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}