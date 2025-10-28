import torch
from torch.utils.data import Dataset
import pickle
import lmdb


class LMDBManager:
    """
    Utility class for reading and caching data from an LMDB database.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self._lmdb_env = None
        self.index_to_key = []
        self._cache = {}
        self._open_lmdb_env()

    def _open_lmdb_env(self):
        """Open LMDB environment and build key index."""
        self._lmdb_env = lmdb.open(
            self.data_path, readonly=True, lock=False, map_size=1e12, subdir=False
        )
        self.index_to_key = []

        with self._lmdb_env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                if key == b"length":
                    continue
                self.index_to_key.append(key)

    def get_data(self, key):
        """Read data by key from LMDB with caching."""
        if key in self._cache:
            return self._cache[key]

        with self._lmdb_env.begin() as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {key} not found in LMDB")
            item = pickle.loads(value)

        self._cache[key] = item
        return item

    def close(self):
        """Close LMDB environment."""
        if self._lmdb_env:
            self._lmdb_env.close()


class CristalDataset(Dataset):
    """
    Dataset for crystal trajectories stored in LMDB format.
    """

    def __init__(self, data_path, training_timestep):
        """
        Args:
            data_path (str): Path to LMDB database.
            training_timestep (int): Number of timesteps for training.
        """
        self.data_path = data_path
        self.training_timestep = training_timestep
        self.lmdb_manager = LMDBManager(data_path)
        self.index_to_key = self.lmdb_manager.index_to_key
        self.length = len(self.index_to_key)
        self.sample_indices = list(range(self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Retrieve one item from the LMDB by index.
        """
        if index < 0 or index >= self.length:
            raise IndexError("Index out of range")

        actual_index = self.sample_indices[index]
        key = self.index_to_key[actual_index]
        item = self.lmdb_manager.get_data(key)

        relaxed = item["relaxed"]
        traj = item["traj"]

        # Extract relaxed structure information
        relaxed_pos = relaxed.get("pos")
        relaxed_cell = relaxed.get("cell")
        relaxed_e = relaxed.get("energy")
        relaxed_atomic_numbers = relaxed.get("atomic_numbers")
        hull_energy = item.get("hull_energy")
        relaxed_num_atoms = torch.tensor([relaxed_pos.shape[0]], dtype=torch.long)

        # Extract trajectory information
        traj_pos = traj.get("pos")
        traj_cell = traj.get("cell")
        traj_e = traj.get("e")
        max_time = traj_pos.shape[0]

        if self.training_timestep > max_time:
            raise ValueError(
                f"Training timestep ({self.training_timestep}) exceeds max time ({max_time}) for key {key}"
            )

        # Uniformly sample timesteps
        sampled_indices = torch.linspace(0, max_time - 1, self.training_timestep).long()
        traj_pos_sampled = traj_pos[sampled_indices]
        traj_cell_sampled = traj_cell[0].unsqueeze(0)
        traj_e_sampled = traj_e[sampled_indices]

        return (
            (relaxed_atomic_numbers, relaxed_cell, relaxed_pos, relaxed_e, relaxed_num_atoms),
            (traj_pos_sampled, traj_cell_sampled, traj_e_sampled),
            key,
            hull_energy,
        )


def custom_collate_fn(batch):
    """
    Custom collate function to batch crystal structures and trajectories.
    """
    batch = [item for item in batch if item is not None]
    relaxed_data, traj_data, key, hull_energy = zip(*batch)

    atomic_numbers = torch.cat([item[0] for item in relaxed_data], dim=0)
    num_atoms = torch.tensor([item[4] for item in relaxed_data])
    relaxed_positions = torch.cat([item[2] for item in relaxed_data], dim=0)
    relaxed_cells = torch.stack([item[1] for item in relaxed_data], dim=0)
    relaxed_energies = torch.tensor([item[3] for item in relaxed_data])

    traj_cell = torch.stack([item[1] for item in traj_data], dim=1)
    traj_pos = torch.cat([item[0] for item in traj_data], dim=1)
    traj_e = torch.stack([item[2] for item in traj_data], dim=1)

    return (
        (relaxed_cells, relaxed_positions, relaxed_energies),
        (traj_pos, traj_cell, traj_e),
        atomic_numbers,
        num_atoms,
        list(key),
        list(hull_energy),
    )


def worker_init_fn(worker_id):
    """
    Initialize LMDBManager for each DataLoader worker.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    if isinstance(dataset, CristalDataset):
        dataset.lmdb_manager = LMDBManager(dataset.data_path)
