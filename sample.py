import os
import glob
import json
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

from train import DSBDiffusionModel, Loss
from data import CristalDataset, custom_collate_fn, worker_init_fn
from utils import get_available_device


class Sampler:
    """
    Sampling class for generating relaxed crystal structures 
    using a pre-trained DSBDiffusionModel.
    """

    def __init__(
        self,
        batch_size: int,
        repeat: int,
        perturb_scale: float,
        data_path: str,
        model_dir: str,
        save_dir: str,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.repeat = repeat
        self.perturb_scale = perturb_scale
        self.data_path = data_path
        self.model_dir = model_dir
        self.save_dir = save_dir

    def load_data(self):
        """Load the test dataset and create a DataLoader."""
        test_set = CristalDataset(self.data_path, self.timesteps)
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            worker_init_fn=worker_init_fn,
        )

    def load_model(self, version: str, config):
        """Load the trained DSBDiffusionModel and its checkpoint."""
        self.model_version = version.split(".")[0]
        self.sample_version = version
        os.makedirs(self.save_dir, exist_ok=True)
        self.timesteps = config.noiser.timesteps

        # Initialize model
        self.dif = DSBDiffusionModel(
            version=self.sample_version,
            train_config=config.train,
            denoiser_config=config.gemnet,
            noiser_config=config.noiser,
        )
        self.device = torch.device(f"cuda:{get_available_device()}")
        self.dif.to(self.device)

        # Find checkpoint file
        checkpoint_pattern = os.path.join(self.model_dir, f"{self.model_version}*.ckpt")
        files = glob.glob(checkpoint_pattern)
        assert files, f"No checkpoint files found for version {self.model_version}."

        latest_ckpt_file = max(files, key=os.path.getmtime)
        print(f"Loading checkpoint from: {latest_ckpt_file}")

        # Load checkpoint
        state_dict = torch.load(latest_ckpt_file, map_location=self.device)
        self.dif.load_state_dict(state_dict["state_dict"])
        self.dif.eval()

        self.loss_fn = Loss(weights=config.train.weights)
        self.timesteps = config.noiser.timesteps

    def get_batch(self, batch):
        """Unpack batch data and move tensors to device."""
        relaxed_data, traj_data, atomic_numbers, num_atoms, keys, hull_energy = batch
        traj_pos, traj_cell, traj_e = traj_data
        relaxed_cells, relaxed_positions, relaxed_energies = relaxed_data

        traj_pos = traj_pos.to(torch.float32).to(self.device)
        traj_cell = traj_cell.to(torch.float32).to(self.device)
        traj_e = traj_e.to(torch.float32).to(self.device)
        atomic_numbers = atomic_numbers.to(self.device)
        num_atoms = num_atoms.to(self.device)

        return (
            atomic_numbers,
            num_atoms,
            traj_pos,
            traj_cell,
            traj_e,
            relaxed_cells,
            relaxed_positions,
            relaxed_energies,
            hull_energy,
            keys,
        )

    def sample(self):
        """Run the sampling process and save results as JSONL."""
        total_batches = len(self.test_loader)
        base_version = self.sample_version.split(".")[0]
        jsonl_save_path = os.path.join(self.save_dir, f"{base_version}_{self.perturb_scale}.jsonl")

        for batch_idx, batch in enumerate(
            tqdm(self.test_loader, total=total_batches, desc="Processing batches")
        ):
            atomic_numbers, num_atoms, traj_pos, traj_cell, traj_e, relaxed_cells, relaxed_positions, relaxed_energies, hull_energy, keys = self.get_batch(batch)

            l = traj_cell[0]
            x_init = traj_pos[0]
            batch_size = l.shape[0]
            cumsum_atoms = torch.cumsum(num_atoms, dim=0)
            shared_cell_tensor = l[0].detach().cpu()
            shared_cell = shared_cell_tensor.tolist()

            all_pred_list = [[] for _ in range(batch_size)]

            for repeat_idx in tqdm(range(self.repeat), desc=f"Batch {batch_idx}: Sampling", leave=False):
                start_time = time.time()

                x = x_init.clone()
                noise_init = torch.randn_like(x, device=self.device)
                x = x + self.perturb_scale * noise_init

                for t in range(0, self.timesteps - 1):
                    noise_x = torch.randn_like(x, device=self.device)
                    x = x + self.dif.noiser.noise_scales[t] * noise_x
                    t_list = torch.full((batch_size,), t, device=self.device)

                    with torch.no_grad():
                        x_vel, e = self.dif.model(a=atomic_numbers, x=x, l=l, n=num_atoms, t=t_list)
                        x = x_vel.squeeze(-1) * (1 / (self.timesteps - 1)) + x

                    del x_vel
                    torch.cuda.empty_cache()

                elapsed_time = time.time() - start_time
                x = x.detach().cpu()
                e = e.detach().cpu()

                for i in range(batch_size):
                    idx_start = cumsum_atoms[i] - num_atoms[i]
                    idx_end = cumsum_atoms[i]
                    pos_pred = x[idx_start:idx_end, :].tolist()
                    e_pred = e[i].view(1).tolist()

                    all_pred_list[i].append(
                        {"index": repeat_idx, "pos": pos_pred, "e": e_pred, "time": elapsed_time}
                    )

            # Write sampled structures to JSONL
            for i in range(batch_size):
                idx_start = cumsum_atoms[i] - num_atoms[i]
                idx_end = cumsum_atoms[i]

                atomic_numbers_data = atomic_numbers[idx_start:idx_end].tolist()

                pos_original = traj_pos[0][idx_start:idx_end, :].detach().cpu().tolist()
                e_original = traj_e[0][i].detach().cpu().view(1).tolist()
                original = {"pos": pos_original, "e": e_original}

                pos_target = traj_pos[-1][idx_start:idx_end, :].detach().cpu().tolist()
                e_target = traj_e[-1][i].detach().cpu().view(1).tolist()
                target = {"pos": pos_target, "e": e_target}

                pos_dft_cart = relaxed_positions[idx_start:idx_end, :].to(torch.float32)
                cell_dft = relaxed_cells[i].to(torch.float32).squeeze(0)
                frac_dft = torch.linalg.solve(cell_dft.T, pos_dft_cart.T).T
                mapped_pos = torch.matmul(frac_dft, shared_cell_tensor.T).T.squeeze(0)
                dft = {"pos": mapped_pos.tolist(), "e": relaxed_energies[i].view(1).tolist()}

                record = {
                    "key": keys[i].decode("ascii"),
                    "hull_energy": hull_energy[i],
                    "atomic_numbers": atomic_numbers_data,
                    "cell": shared_cell,
                    "original": original,
                    "target": target,
                    "dft": dft,
                    "pred_list": all_pred_list[i],
                }

                with open(jsonl_save_path, "a", encoding="utf-8") as jsonl_file:
                    json.dump(record, jsonl_file, ensure_ascii=False)
                    jsonl_file.write("\n")


def main(version, config_path="config.yaml"):
    """Entry point for sampling process."""
    config = OmegaConf.load(config_path)
    sampler: Sampler = hydra.utils.instantiate(config.sampler)
    sampler.load_model(version, config)
    sampler.load_data()
    sampler.sample()
