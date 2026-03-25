import torch
from torch.utils.data import Dataset

from zoorvival.data import TCGADataSplit

__all__ = ["TCGADataset", "as_torch_dataset"]


class TCGADataset(Dataset):
    def __init__(self, data_split: TCGADataSplit):
        self.data_split = data_split

    def __len__(self) -> int:
        return len(self.data_split.df_clinical)

    def __getitem__(self, idx: int):
        clinical = torch.as_tensor(
            self.data_split.df_clinical.iloc[idx].values, dtype=torch.float32
        )
        cnv = torch.as_tensor(self.data_split.df_cnv.iloc[idx].values, dtype=torch.float32)
        dnam = torch.as_tensor(
            self.data_split.df_dnam.iloc[idx].values, dtype=torch.float32
        )
        mirna = torch.as_tensor(
            self.data_split.df_mirna.iloc[idx].values, dtype=torch.float32
        )
        mrna = torch.as_tensor(self.data_split.df_mrna.iloc[idx].values, dtype=torch.float32)
        wsi_embeddings = torch.as_tensor(
            self.data_split.wsi_embeddings[idx], dtype=torch.float32
        )
        event = torch.tensor(self.data_split.y[idx]["event"], dtype=torch.float32)
        time = torch.tensor(self.data_split.y[idx]["time"], dtype=torch.float32)

        return (
            clinical,
            cnv,
            dnam,
            mirna,
            mrna,
            wsi_embeddings,
            event,
            time,
        )


def as_torch_dataset(
    data_split: TCGADataSplit,
) -> TCGADataset:
    """Convert a TCGADataSplit to a PyTorch Dataset."""
    return TCGADataset(data_split)
