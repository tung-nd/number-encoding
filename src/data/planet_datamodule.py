from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from lightning import LightningDataModule

### Define collator and data loaders
def define_masked_num_collator(pad_token_id, mask_token_id, mlm_probability):
    def masked_num_collator(batch):
        x = [torch.tensor(sample["input_ids"]) for sample in batch]
        x_num = [torch.tensor(sample["numbers"]) for sample in batch]
        x = pad_sequence(x, batch_first=True, padding_value=pad_token_id)
        x_num = pad_sequence(x_num, batch_first=True, padding_value=1)
        probability_matrix = torch.full(x.shape, mlm_probability)
        mask = torch.bernoulli(probability_matrix).bool()
        y = x.clone()
        y_num = x_num.clone()
        y[~mask] = -100 # so that unmasked tokens are not used for loss calculation
        x[mask] = mask_token_id # replace actual tokens with mask token
        x_num[mask] = 1 # replace actual number values with 1
        return {"x": x, "x_num": x_num, "y": y, "y_num": y_num, "mask": mask}

    return masked_num_collator

class PlanetDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path,
        tokenizer_path,
        mlm_probability=0.3,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=64,
        num_workers=1,
        pin_memory=False,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            tokenized_ds = Dataset.load_from_disk(self.hparams.dataset_path) # all data
            # split train and val+test
            tokenized_ds = tokenized_ds.train_test_split(test_size=self.hparams.test_ratio+self.hparams.val_ratio)
            train_ds, test_ds = tokenized_ds["train"], tokenized_ds["test"]
            # split val and test
            test_ds = test_ds.train_test_split(
                test_size=self.hparams.test_ratio/(self.hparams.test_ratio+self.hparams.val_ratio)
            )
            val_ds, test_ds = test_ds["train"], test_ds["test"]
            
            self.data_train = train_ds
            self.data_val = val_ds
            self.data_test = test_ds
            
            # tokenizer to create collator
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=self.hparams.tokenizer_path,
                bos_token="[END]",
                eos_token="[END]",
                mask_token="[MASK]",
                pad_token="[PAD]",
            )
            pad_token_id = tokenizer.pad_token_id
            mask_token_id = tokenizer.mask_token_id
            self.collator = define_masked_num_collator(pad_token_id, mask_token_id, self.hparams.mlm_probability)
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collator,
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collator,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collator,
        )
        
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