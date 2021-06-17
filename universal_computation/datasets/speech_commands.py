from torch.utils.data import DataLoader
import torchaudio


from universal_computation.datasets.dataset import Dataset
from universal_computation.datasets.helpers.speech_commands import collate_fn, SubsetSC


class SpeechCommandsDataset(Dataset):

    def __init__(self, batch_size, patch_size, sample_rate=8000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader
        self.patch_size = patch_size
        self.sample_rate = sample_rate

        self.transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=sample_rate)

        train_set = SubsetSC("training")
        test_set = SubsetSC("testing")

        self.d_train = DataLoader(
            train_set, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, shuffle=True,
        )
        self.d_test = DataLoader(
            test_set, batch_size=batch_size,  collate_fn=collate_fn, drop_last=True, shuffle=True,
        )

        self.train_iter = iter(self.d_train)
        self.test_iter = iter(self.d_test)

    def get_batch(self, batch_size=None, train=True):
        if train:
            x, y = next(self.train_iter, (None, None))
            if x is None:
                self.train_iter = iter(self.d_train)
                x, y = next(self.train_iter)
        else:
            x, y = next(self.test_iter, (None, None))
            if x is None:
                self.test_iter = iter(self.d_test)
                x, y = next(self.test_iter)

        x = self.transform(x)

        x = x.reshape(self.batch_size, self.sample_rate//self.patch_size, self.patch_size)

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x, y