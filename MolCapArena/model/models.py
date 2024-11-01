import os
from typing import List, Dict, Callable, Union, Tuple

import lightning as L
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from MolCapArena.data.processing import PropTask
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch import optim, nn, utils, Tensor
from transformers import AutoConfig, AutoModel
import torchmetrics

from MolCapArena.model.components import MLP, GNNMolEncoder


class GNNCaptionEmbModel(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()

        self.config = config
        self.out_channels = output_dim

        self.mol_encoder = GNNMolEncoder(
            node_dim=self.config["node_dim"],
            edge_dim=self.config["edge_dim"],
            hidden_dim_graph=self.config["hidden_dim_graph"],
            hidden_dim_ffn=None,
            num_mp_layers=self.config["num_mp_layers"],
            out_channels=config["latent_size"],
            dropout=self.config["dropout"],
            num_readout_layers=1,
            mol_features_size=0,
            aggr=self.config["aggr"],
            jk=self.config["jk"],
        )

        self.output_dim = output_dim

        self.text_proj = MLP(
            input_dim=config["text_latent_size"],
            hidden_dim=[config["latent_size"], config["latent_size"]],
            output_dim=config["latent_size"],
        )

        self.combine = MLP(
            input_dim=2 * config["latent_size"],
            hidden_dim=[config["latent_size"], config["latent_size"]],
            output_dim=output_dim,
        )

    def forward(self, batch, get_embedding=False):
        mol_input = batch["input"]["molecule"]
        mol_emb = self.mol_encoder(mol_input)

        text_emb = batch["input"]["caption_emb"]
        text_emb = self.text_proj(text_emb)

        x = torch.cat((text_emb, mol_emb), dim=1)
        if get_embedding:
            pred = self.combine.forward_embedding(x)
        else:
            pred = self.combine(x)

        return pred

    def freeze_text_model(self):
        for p in self.text_encoder_model.parameters():
            p.requires_grad = False

    def freeze_mol_model(self):
        for name, p in self.mol_encoder.named_parameters():
            if name == "classifier.1.weight" or name == "classifier.1.bias":
                continue
            p.requires_grad = False


class GNNCaptionModel(nn.Module):
    def __init__(self, config, output_dim, text_label="caption"):
        super().__init__()

        self.config = config
        self.out_channels = output_dim

        self.mol_encoder = GNNMolEncoder(
            node_dim=self.config["node_dim"],
            edge_dim=self.config["edge_dim"],
            hidden_dim_graph=self.config["hidden_dim_graph"],
            hidden_dim_ffn=None,
            num_mp_layers=self.config["num_mp_layers"],
            out_channels=config["latent_size"],
            dropout=self.config["dropout"],
            num_readout_layers=1,
            mol_features_size=0,
            aggr=self.config["aggr"],
            jk=self.config["jk"],
        )

        self.output_dim = output_dim

        self.text_encoder_model = AutoModel.from_pretrained(
            config["pretrained_text_model"]
        )
        self.text_config = AutoConfig.from_pretrained(config["pretrained_text_model"])
        self.text_label = text_label

        self.text_proj = MLP(
            input_dim=self.text_config.hidden_size,
            hidden_dim=[config["latent_size"], config["latent_size"]],
            output_dim=config["latent_size"],
        )

        self.combine = MLP(
            input_dim=2 * config["latent_size"],
            hidden_dim=[config["latent_size"], config["latent_size"]],
            output_dim=output_dim,
        )

    def forward(self, batch, get_embedding=False):
        mol_input = batch["input"]["molecule"]
        mol_emb = self.mol_encoder(mol_input)

        text_input = batch["input"][self.text_label]
        text_emb = self.text_encoder_model(**text_input)["pooler_output"]
        text_emb = self.text_proj(text_emb)

        x = torch.cat((text_emb, mol_emb), dim=1)
        if get_embedding:
            pred = self.combine.forward_embedding(x)
        else:
            pred = self.combine(x)

        return pred

    def freeze_text_model(self):
        for p in self.text_encoder_model.parameters():
            p.requires_grad = False

    def freeze_mol_model(self):
        for name, p in self.mol_encoder.named_parameters():
            if name == "classifier.1.weight" or name == "classifier.1.bias":
                continue
            p.requires_grad = False


class GNNOnlyModel(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()

        self.config = config
        self.out_channels = output_dim

        self.mol_encoder = GNNMolEncoder(
            node_dim=self.config["node_dim"],
            edge_dim=self.config["edge_dim"],
            hidden_dim_graph=self.config["hidden_dim_graph"],
            hidden_dim_ffn=None,
            num_mp_layers=self.config["num_mp_layers"],
            out_channels=output_dim,
            dropout=self.config["dropout"],
            num_readout_layers=1,
            mol_features_size=0,
            aggr=self.config["aggr"],
            jk=self.config["jk"],
        )

    def forward(self, batch, get_embedding=False):
        mol_input = batch["input"]["molecule"]
        if get_embedding:
            graph_repr = self.mol_encoder.compute_message_passing(mol_input)
            pred = self.mol_encoder.compute_readout(
                graph_repr, restrict_output_layers=-1
            )
        else:
            pred = self.mol_encoder(mol_input)

        return pred


class CaptionOnlyModel(nn.Module):
    def __init__(self, config, output_dim, text_label="caption"):
        super().__init__()

        self.config = config

        self.text_encoder_model = AutoModel.from_pretrained(
            config["pretrained_text_model"]
        )
        self.text_config = AutoConfig.from_pretrained(config["pretrained_text_model"])
        self.text_label = text_label
        self.output_dim = output_dim
        self.out_channels = output_dim

        self.text_proj = MLP(
            input_dim=self.text_config.hidden_size,
            hidden_dim=[config["latent_size"], config["latent_size"]],
            output_dim=output_dim,
        )

    def forward(self, batch, get_embedding=False):
        text_input = batch["input"][self.text_label]

        text_emb = self.text_encoder_model(**text_input)["pooler_output"]

        if get_embedding:
            pred = self.text_proj.forward_embedding(text_emb)
        else:
            pred = self.text_proj(text_emb)

        return pred

    def freeze_text_model(self):
        for p in self.text_encoder_model.parameters():
            p.requires_grad = False


class MoleculeTextModel(L.LightningModule):
    def __init__(self, config, encoder, **kwargs):
        super().__init__()
        self.config = config

        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder
        self.task = PropTask(self.hparams.task)
        if self.task not in (PropTask.BINARY_CLASSIFICATION, PropTask.REGRESSION):
            raise ValueError(f"Unsupported task: {self.task}")

        self.loss_module = self.get_loss_func()
        self.metrics = self.get_metrics()

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.mask_y_nan = False

        self.pred_embs = False

    def get_metrics(
        self,
    ) -> Dict[str, Callable[[torch.tensor, torch.tensor], torch.tensor]]:
        metrics = dict()
        if self.task is PropTask.BINARY_CLASSIFICATION:
            metrics["auc"] = torchmetrics.classification.BinaryAUROC()
            metrics["auprc"] = torchmetrics.classification.BinaryAveragePrecision()
            metrics["accuracy"] = torchmetrics.classification.BinaryAccuracy()
            metrics["precision"] = torchmetrics.classification.BinaryPrecision()
            metrics["recall"] = torchmetrics.classification.BinaryRecall()

        elif self.task is PropTask.REGRESSION:
            metrics["mse"] = torchmetrics.regression.MeanSquaredError(
                num_outputs=self.encoder.out_channels
            )
            metrics["mae"] = torchmetrics.regression.MeanAbsoluteError()
            metrics["r2"] = torchmetrics.regression.R2Score(
                num_outputs=self.encoder.out_channels
            )
            metrics["pearson_coef"] = torchmetrics.regression.PearsonCorrCoef(
                num_outputs=self.encoder.out_channels
            )
            metrics["spearman_coef"] = torchmetrics.regression.SpearmanCorrCoef(
                num_outputs=self.encoder.out_channels
            )

        elif self.task is PropTask.MULTI_CLASSIFICATION:
            metrics["auc"] = torchmetrics.classification.MulticlassAUROC(
                num_classes=self.hparams["num_classes"]
            )
            metrics["auprc"] = torchmetrics.classification.MulticlassAveragePrecision(
                num_classes=self.hparams["num_classes"]
            )
            metrics["accuracy"] = torchmetrics.classification.MulticlassAccuracy(
                num_classes=self.hparams["num_classes"]
            )
            metrics["precision"] = torchmetrics.classification.MulticlassPrecision(
                num_classes=self.hparams["num_classes"]
            )
            metrics["recall"] = torchmetrics.classification.MulticlassRecall(
                num_classes=self.hparams["num_classes"]
            )

        return metrics

    def get_loss_func(self):
        if self.task is PropTask.BINARY_CLASSIFICATION:
            loss_module = nn.BCEWithLogitsLoss()
        elif self.task is PropTask.REGRESSION:
            loss_module = nn.MSELoss()
        elif self.task is PropTask.MULTI_CLASSIFICATION:
            if self.hparams.class_weight is None:
                loss_module = nn.CrossEntropyLoss()
            else:
                loss_module = nn.CrossEntropyLoss(
                    weight=torch.tensor(self.hparams.class_weight)
                )
        return loss_module

    def forward(self, batch):
        o = self._compute_with_activation(batch, return_logits=False)
        return o

    def _compute_with_activation(
        self, batch, return_logits: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        logits = self.encoder(batch)

        if self.task is PropTask.BINARY_CLASSIFICATION:
            o = torch.sigmoid(logits)
        elif self.task is PropTask.REGRESSION:
            o = F.relu(logits) if self.hparams.final_relu else logits
        elif self.task is PropTask.MULTI_CLASSIFICATION:
            o = torch.softmax(logits, dim=1)
        else:
            raise ValueError

        return o, logits if return_logits else o

    def _prepare_labels(self, targets: torch.Tensor) -> torch.Tensor:
        if self.task is PropTask.MULTI_CLASSIFICATION:
            targets = targets.long()

        return targets

    def _prepare_labels_return(self, targets: torch.Tensor) -> torch.Tensor:
        if self.task in (
            PropTask.BINARY_CLASSIFICATION,
            PropTask.MULTI_CLASSIFICATION,
        ):
            targets = targets.long()
        return targets

    def _log_metric(self, step_name, preds, y, task_name=None):
        for metric_name, metric_func in self.metrics.items():
            if task_name is not None:
                metric_str = f"{step_name}/{task_name}-{metric_name}"
            else:
                metric_str = f"{step_name}/{metric_name}"
            self.log(
                metric_str,
                metric_func(preds, y).mean().item(),
                prog_bar=True,
                sync_dist=True,
            )

    def _compute_masked_loss(self, logits, labels):
        # Create a mask to identify where labels are not NaN
        mask = ~torch.isnan(labels)
        # Select only the elements corresponding to non-NaN labels
        valid_logits = torch.masked_select(logits, mask)
        valid_label = torch.masked_select(labels, mask)

        # Use loss function to compute the loss on the valid elements
        valid_logits = valid_logits.to(torch.float32)
        valid_label = valid_label.to(torch.float32)

        loss = self.loss_module(
            valid_logits,
            valid_label,
        )
        return loss, mask

    def compute_step(
        self,
        batch,
        prefix: str,
    ) -> torch.Tensor:
        # 1. compute loss
        y = batch["input"]["activity"]
        y = self._prepare_labels(y)

        outputs, logits = self._compute_with_activation(batch, return_logits=True)
        batch_size = y.shape[0]

        if PropTask.MULTI_CLASSIFICATION != self.task and (
            (y.shape[1] > 1) or (self.mask_y_nan is True)
        ):  # If there are multiple tasks
            loss, mask = self._compute_masked_loss(logits, y)
            # clean probs and labels
            outputs, y = torch.masked_select(outputs, mask), torch.masked_select(
                y, mask
            )
        else:
            loss = self.loss_module(logits, y)
        self.log(
            f"{prefix}/loss",
            loss.item(),
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # 3. prepare y for logging
        y = self._prepare_labels_return(y)

        if prefix == "train":
            self._log_metric(step_name=prefix, preds=outputs, y=y)

        return {"loss": loss, "probs": outputs.detach(), "labels": y}

    def on_validation_epoch_end(self):
        all_validation_probs = torch.cat(
            [i["probs"] for i in self.validation_step_outputs]
        )
        all_validation_labels = torch.cat(
            [i["labels"] for i in self.validation_step_outputs]
        )
        self._log_metric("val", all_validation_probs, all_validation_labels)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        all_test_probs = torch.cat([i["probs"] for i in self.test_step_outputs])
        all_test_labels = torch.cat([i["labels"] for i in self.test_step_outputs])
        self._log_metric("test", all_test_probs, all_test_labels)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self.compute_step(batch, prefix="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        step_outputs = self.compute_step(batch, prefix="val")
        self.validation_step_outputs.append(step_outputs)
        return step_outputs

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        step_outputs = self.compute_step(batch, prefix="test")
        self.test_step_outputs.append(step_outputs)
        return step_outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # don't use self.trainer.num_training_batches as it is inf here
        # num_train_batches = self.hparams.num_train_batches
        num_training_steps = self.trainer.estimated_stepping_batches

        scheduler_warmup_cosine = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.config["num_warmup_steps"],
            max_epochs=num_training_steps,
            eta_min=self.hparams.lr / 10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler_warmup_cosine, "interval": "step"},
        }

    def on_fit_start(self):
        # fix metrics devices
        for k, v in self.metrics.items():
            self.metrics[k] = v.to(self.device)

    def on_validation_start(self):
        # fix metrics devices
        for k, v in self.metrics.items():
            self.metrics[k] = v.to(self.device)

    def on_test_start(self):
        # fix metrics devices
        for k, v in self.metrics.items():
            self.metrics[k] = v.to(self.device)

    def predict_step(self, batch) -> torch.Tensor:
        if not self.pred_embs:
            out = self(batch)
            return out, batch
        else:
            return self.encoder(batch, get_embedding=True), batch

    def predict_data(self, dataloader, gpus: int = 1):
        trainer = L.Trainer(devices=gpus, logger=False, accelerator="gpu")
        res = trainer.predict(self, dataloaders=dataloader)
        res = [i.cpu().numpy() for i in res[0]]
        res = np.concatenate(res)

        return res
