from typing import Optional

import os
import math
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from lightning import LightningDataModule

### Define collator and data loaders
def define_masked_num_collator(pad_token_id, mask_token_id, dataset_name):
    def masked_num_collator(batch):
        x = [torch.tensor(sample["input_ids"]) for sample in batch]
        x_num = [torch.tensor(sample["numbers"]).float() for sample in batch]
        x = pad_sequence(x, batch_first=True, padding_value=pad_token_id)
        x_num = pad_sequence(x_num, batch_first=True, padding_value=1)
        
        # mask is always the last token for this task
        mask = torch.zeros_like(x)
        mask[:, -1] = 1
        mask = mask.bool()
        
        y = x.clone()
        y_num = x_num.clone()
        y[~mask] = -100 # so that unmasked tokens are not used for loss calculation
        x[mask] = mask_token_id # replace actual tokens with mask token
        x_num[mask] = 1 # replace actual number values with 1
        return {"x": x, "x_num": x_num, "y": y, "y_num": y_num, "mask": mask, "dataset_name": dataset_name}

    return masked_num_collator

class MultiDataModule(LightningDataModule):
    def __init__(
        self,
        data_root,
        tokenizer_path,
        num_digit=5,
        batch_size=64,
        num_workers=1,
        pin_memory=False,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        max_value = int(math.pow(10, num_digit)) - 1  # Maximum multiplicand value based on num_digit
        self.product_min_value = 1  # Minimum product value
        self.product_max_value = (max_value ** 2)  # Maximum possible product value
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def denormalize(self, normalized_number):
        """Denormalize a number from the range [-5, 5] back to its original scale."""
        # return (normalized_number + 5) / 10 * (self.product_max_value - self.product_min_value) + self.product_min_value
        log_min_val = math.log(self.product_min_value)
        log_max_val = math.log(self.product_max_value)
        log_value = normalized_number * (log_max_val - log_min_val) + log_min_val
        return torch.exp(log_value)
        
    def setup(self, stage: Optional[str] = None) -> None:
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            data_train = Dataset.load_from_disk(os.path.join(self.hparams.data_root, "multi_train_tokenized"))
            
            data_val = {}
            data_test = {}
            digits = list(range(1, self.hparams.num_digit + 1))
            for a_num_digit in digits:
                for b_num_digit in digits[:a_num_digit]:
                    val_name = f'multi_val_{a_num_digit}_by_{b_num_digit}_tokenized'
                    data_val[f'{a_num_digit}_by_{b_num_digit}'] = Dataset.load_from_disk(os.path.join(self.hparams.data_root, val_name))
                    
                    test_name = f'multi_test_{a_num_digit}_by_{b_num_digit}_tokenized'
                    data_test[f'{a_num_digit}_by_{b_num_digit}'] = Dataset.load_from_disk(os.path.join(self.hparams.data_root, test_name))
            
            self.data_train = data_train
            self.data_val = data_val
            self.data_test = data_test
            
            # tokenizer to create collator
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=self.hparams.tokenizer_path,
                bos_token="[END]",
                eos_token="[END]",
                mask_token="[MASK]",
                pad_token="[PAD]",
            )
            self.pad_token_id = tokenizer.pad_token_id
            self.mask_token_id = tokenizer.mask_token_id
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=define_masked_num_collator(self.pad_token_id, self.mask_token_id, "train"),
        )
        
    def val_dataloader(self):
        dataloaders = {
            val_name: DataLoader(
                self.data_val[val_name],
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                collate_fn=define_masked_num_collator(self.pad_token_id, self.mask_token_id, val_name),
            )
            for val_name in self.data_val
        }
        return dataloaders
        
    def test_dataloader(self):
        dataloaders = {
            test_name: DataLoader(
                self.data_test[test_name],
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                collate_fn=define_masked_num_collator(self.pad_token_id, self.mask_token_id, test_name),
            )
            for test_name in self.data_test
        }
        return dataloaders
        
# datamodule = PlanetDataModule(
#     dataset_path="data/tokenized_ds_all",
#     tokenizer_path="tokenizer.json",
#     mlm_probability=0.3,
#     train_ratio=0.8,
#     val_ratio=0.1,
#     test_ratio=0.1,
#     batch_size=64,
#     num_workers=1,
#     pin_memory=False,
# )
# datamodule.setup()
# print (len(datamodule.data_train), len(datamodule.data_val), len(datamodule.data_test))
# print (datamodule.data_train[0])