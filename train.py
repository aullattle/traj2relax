from omegaconf import OmegaConf
from noiser import Noiser
from utils import XlsxDataWriter
from traj2relax import Traj2Relax
import hydra
from data import CristalDataset, custom_collate_fn, worker_init_fn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import AdamW
import torch
import os
from pytorch_lightning.strategies import DDPStrategy


class Loss:
    """
    Custom loss class with flexible weighted component control.
    """

    def __init__(self, weights):
        self.weights = OmegaConf.to_container(weights, resolve=True)
        self.loss_fn = nn.MSELoss()
        self.loss_dict = {}
        self.accumulated_loss = {key: 0.0 for key in self.weights}
        self.batch_count = 0

    def __call__(self, outputs, targets):
        """
        Compute weighted loss for each component.
        """
        if set(outputs.keys()) != set(targets.keys()):
            raise ValueError("Outputs and targets must have the same keys.")
        if not all(key in outputs for key in self.weights):
            raise ValueError("Some keys in weights are missing from outputs/targets.")

        self.loss_dict.clear()
        aggregated_loss = 0.0

        for key, weight in self.weights.items():
            if key == 'pos':
                outputs[key] = outputs[key].squeeze(-1)
            loss = self.loss_fn(outputs[key].float(), targets[key].float())
            weighted_loss = loss * weight
            aggregated_loss += weighted_loss

            self.accumulated_loss[key] += weighted_loss.detach().item()
            self.loss_dict[key] = weighted_loss.detach().item()

        # Ensure loss is finite
        if not torch.isfinite(aggregated_loss):
            return aggregated_loss

        self.batch_count += 1
        return aggregated_loss

    def get_no_weight_dict_loss(self, outputs, targets):
        """
        Compute unweighted loss dictionary for each component.
        """
        if set(outputs.keys()) != set(targets.keys()):
            raise ValueError("Outputs and targets must have the same keys.")
        if not all(key in outputs for key in self.weights):
            raise ValueError("Some keys in weights are missing from outputs/targets.")

        current_dict = {}
        for key in self.weights:
            loss = self.loss_fn(outputs[key].float(), targets[key].float())
            current_dict[key] = loss.item()
        return current_dict

    def get_average_loss(self):
        """
        Return average accumulated loss per component and reset counters.
        """
        if self.batch_count == 0:
            return {key: 0.0 for key in self.weights}

        avg_loss = {key: self.accumulated_loss[key] / self.batch_count for key in self.weights}
        self.accumulated_loss = {key: 0.0 for key in self.weights}
        self.batch_count = 0
        return avg_loss


class DSBDiffusionModel(pl.LightningModule):
    """
    Diffusion model based on Traj2Relax with custom training loop.
    """

    def __init__(self, version, train_config, denoiser_config, noiser_config):
        super().__init__()
        self.automatic_optimization = False

        self.data_path = train_config.data_path
        self.save_path = train_config.save_path
        self.local_result_path = train_config.local_result_path
        self.lr = train_config.lr
        self.max_lr = train_config.max_lr
        self.weight_decay = train_config.weight_decay
        self.batch_size = train_config.batch_size
        self.epoch = train_config.epoch
        self.save_per_epoch = train_config.save_per_epoch
        self.log_loss_per_step = train_config.log_loss_per_step
        self.loss_fn = Loss(weights=train_config.weights)

        self.noiser: Noiser = hydra.utils.instantiate(noiser_config)
        self.timesteps = self.noiser.timesteps

        self.model = Traj2Relax(noiser=self.noiser, denoiser_config=denoiser_config)

        os.makedirs(os.path.join(train_config.local_result_path, version), exist_ok=True)
        self.local_train_loss_logger = XlsxDataWriter(os.path.join(train_config.local_result_path, version, "train.xlsx"))
        self.local_val_loss_logger = XlsxDataWriter(os.path.join(train_config.local_result_path, version, "val.xlsx"))

        self.prev_params = None
        self.loss_sum = []

    def setup(self, stage):
        train_set_path = self.data_path + "/train.lmdb"
        valid_set_path = self.data_path + "/val.lmdb"
        self.train_set = CristalDataset(train_set_path, self.timesteps)
        self.valid_set = CristalDataset(valid_set_path, self.timesteps)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            worker_init_fn=worker_init_fn,
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        max_epochs = self.trainer.max_epochs
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def get_batch(self, batch: list, batch_idx: int, mode: int):
        """
        Process and extract required tensors for current batch.
        """
        _, traj_data, atomic_numbers, num_atoms, key, _ = batch
        traj_pos, traj_cell, traj_e = traj_data

        with torch.no_grad():
            batch_size = num_atoms.size(0)
            t = torch.randint(0, self.noiser.timesteps, (batch_size,), device=atomic_numbers.device)
            cumsum_atoms = torch.cumsum(num_atoms, dim=0)

            x_mid = torch.cat([traj_pos[t[i], cumsum_atoms[i] - num_atoms[i]:cumsum_atoms[i], :] for i in range(batch_size)], dim=0)
            l_mid = torch.stack([traj_cell[0, i] for i in range(batch_size)])
            e_mid = torch.stack([traj_e[t[i], i] for i in range(batch_size)])

            t_prev = torch.clamp(t + 1, max=self.timesteps - 1)
            x_mid_prev = torch.cat([traj_pos[t_prev[i], cumsum_atoms[i] - num_atoms[i]:cumsum_atoms[i], :] for i in range(batch_size)], dim=0)

            x_t, x_target, l_t, e_target = self.noiser(
                l_mid=l_mid,
                x_mid=x_mid,
                x_mid_prev=x_mid_prev,
                e_mid=e_mid,
                t=t,
                num_atoms=num_atoms,
            )
            a_t = atomic_numbers

        if torch.isnan(l_t).any():
            raise ValueError(f"l_t contains NaN at batch_idx={batch_idx}")

        return a_t, x_t, x_target, l_t, e_target, num_atoms, t

    def training_step(self, batch, batch_idx):
        a_t, x_t, x_target, l_t, e_target, num_atoms, t = self.get_batch(batch, batch_idx, 0)
        pred_x_v, pred_e = self.model(a=a_t, l=l_t, x=x_t, n=num_atoms, t=t)
        pred = {"pos": pred_x_v, "e": pred_e}
        target = {"pos": x_target, "e": e_target}
        loss = self.loss_fn(pred, target)

        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True)

        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if self.global_step % self.log_loss_per_step == 0 and self.global_step != 0:
            avg_loss_dict = self.loss_fn.get_average_loss()
            avg_loss_dict["step"] = self.global_step
            self.local_train_loss_logger.write_dict(avg_loss_dict)

        if self.global_step % 200 == 0:
            torch.cuda.empty_cache()

        return loss.detach()

    def validation_step(self, batch, batch_idx):
        a_t, x_t, x_target, l_t, e_target, num_atoms, t = self.get_batch(batch, batch_idx, 0)
        pred_x_v, pred_e = self.model(a=a_t, l=l_t, x=x_t, n=num_atoms, t=t)
        pred = {"pos": pred_x_v, "e": pred_e}
        target = {"pos": x_target, "e": e_target}
        loss = self.loss_fn(pred, target)
        self.log("val_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            avg_loss_dict = self.loss_fn.get_average_loss()
            avg_loss_dict["step"] = self.global_step
            self.local_val_loss_logger.write_dict(avg_loss_dict)


def main(version, config_path="config.yaml"):
    from pytorch_lightning.callbacks import ModelCheckpoint

    config = OmegaConf.load(config_path)
    dif = DSBDiffusionModel(
        version=version,
        train_config=config.train,
        denoiser_config=config.gemnet,
        noiser_config=config.noiser,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.train.save_path,
        filename=version + "{epoch}-{val_loss:.2f}",
        save_top_k=2,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=config.train.epoch,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=[1],
        precision=16,
    )

    trainer.fit(dif, ckpt_path=None)


if __name__ == "__main__":
    pass
