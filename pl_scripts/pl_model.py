import pytorch_lightning as pl
import torch
from omegaconf import DictConfig


class MultipleChoiceLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        probs = torch.softmax(outputs.logits, dim=1)
        top3 = torch.topk(probs, k=3, dim=1).indices
        labels = batch["labels"].unsqueeze(1)
        hits = (top3 == labels).float()
        inv_ranks = 1.0 / torch.arange(1, 4, device=probs.device).float()
        ap = (hits * inv_ranks.unsqueeze(0)).sum(dim=1)
        map3 = ap.mean()
        self.log(
            "val_map3", map3, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(**batch)
        return outputs.logits.detach().cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.model.learning_rate,
            weight_decay=self.cfg.model.weight_decay,
        )
        return optimizer
