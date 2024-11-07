import torch
from torch.utils.data import Dataset


class QuijoteSeqDataset(Dataset):
    def __init__(self, text, window_size=100, is_for_train=True):
        self.text = self.create_sequences(text, window_size=window_size)
        self.is_for_train = is_for_train

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if self.is_for_train:
            return torch.tensor(self.text[idx][:-1]), torch.tensor(
                self.text[idx][-1]
            )
        else:
            return torch.tensor(self.text[idx])

    @staticmethod
    def create_sequences(text, window_size=3):
        text_windows = []
        for i in range(len(text) - window_size + 1):
            text_windows.append(text[i : i + window_size])
        return text_windows
