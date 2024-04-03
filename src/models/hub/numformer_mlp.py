from torch import nn


class NumformerMLP(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        nhead=6,
        num_layers=6,
        dim_feedforward=3072,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-05,
        batch_first=True,
        norm_first=True,
        transformer_bias=False,
        numhead_bias=True,
        context_length=1024,
        is_causal=False,
    ):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            # bias=transformer_bias,
        )
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=encoder, num_layers=num_layers, enable_nested_tensor=False
        )
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.number_embed = nn.Sequential(
            nn.Linear(1, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.position_embed = nn.Embedding(context_length, d_model)
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=transformer_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, vocab_size, bias=transformer_bias),
        )
        self.num_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=numhead_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1, bias=numhead_bias),
        )
        self.is_causal = is_causal
    
    def set_num_id(self, num_id):
        self.num_id = num_id

    def forward(self, x, x_num):
        token_embeddings = self.token_embed(x)
        number_embeddings = self.number_embed(x_num.unsqueeze(-1))
        # x embeddings equal to token_embeddings when token id is different from self.num_id
        # otherwise, x embeddings equal to number_embeddings
        x = token_embeddings * (x != self.num_id).unsqueeze(-1) + number_embeddings * (x == self.num_id).unsqueeze(-1)
        x = x + self.position_embed.weight[: x.shape[1]].unsqueeze(0)
        x = self.encoder_stack(x, is_causal=self.is_causal)
        logit_preds = self.lm_head(x)
        num_preds = self.num_head(x)
        return logit_preds, num_preds


# from src.data.planet_datamodule import PlanetDataModule
# datamodule = PlanetDataModule(
#     dataset_path='dataset/tokenized_ds_all',
#     tokenizer_path='tokenizer.json',
#     mlm_probability=0.3,
#     train_ratio=0.8,
#     val_ratio=0.1,
#     test_ratio=0.1,
#     batch_size=1,
#     num_workers=1,
#     pin_memory=False,
# )
# datamodule.setup()
# train_loader = datamodule.train_dataloader()
# batch = next(iter(train_loader))
# # print (batch['x'].shape, batch['x_num'].shape, batch['y'].shape, batch['y_num'].shape)
# # print (batch['x'][0])
# # print (batch['x_num'][0])

# model = NumformerMLP(vocab_size=27).cuda()
# model.set_num_id(3)
# logit_preds, num_preds = model(batch['x'].cuda(), batch['x_num'].float().cuda())