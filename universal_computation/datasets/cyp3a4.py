from universal_computation.datasets.dataset import Dataset
from universal_computation.datasets.helpers.cyp3a4 import ChembertaFeaturizer, ChembertaConfig, ChembertaBatchEncoding


class Cyp3A4Dataset(Dataset):

    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader

        from tdc.single_pred import ADME
        data = ADME(name='CYP3A4_Veith')
        split = data.get_split(method='scaffold', seed=0, frac=[0.8, 0.2, 0])

        featurizer = ChembertaFeaturizer(ChembertaConfig(pretrained_name='seyonec/ChemBERTa_zinc250k_v2_40k'))

        train_data = featurizer.encode_smiles_list(split['train']['Drug'], split['train']['Y'])
        test_data = featurizer.encode_smiles_list(split['valid']['Drug'], split['valid']['Y'])

        self.train_loader = featurizer.get_data_loader(train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = featurizer.get_data_loader(test_data, batch_size=batch_size, shuffle=False)

        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)

    def get_batch(self, batch_size=None, train=True):
        if train:
            batch = next(self.train_iter, None)
            if batch is None:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)
        else:
            batch = next(self.test_iter, None)
            if batch is None:
                self.test_iter = iter(self.test_loader)
                batch = next(self.test_iter)

        batch: ChembertaBatchEncoding
        x = batch.data['input_ids'].to(device=self.device)
        y = batch.y.long().view(-1).to(device=self.device)

        self._ind += 1

        return x, y
