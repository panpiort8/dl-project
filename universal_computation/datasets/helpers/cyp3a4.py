from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import transformers

try:
    from huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
    from huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin, RecursiveToDeviceMixin
    from huggingmolecules.featurization.featurization_common_utils import stack_y_list
except ImportError:
    raise ImportError('Please install huggingmolecules from https://github.com/gmum/huggingmolecules. '
                      'Due to python versions conflict, you must remove a `Protocol` occurrence '
                      'from the huggingmolecules src/huggingmolecules/featurization/featurization_api.py file')


@dataclass
class ChembertaConfig(PretrainedConfigMixin):
    pretrained_name: str = None

    @classmethod
    def from_pretrained(cls, pretrained_name: str, **kwargs):
        return cls(pretrained_name=pretrained_name)


@dataclass
class ChembertaBatchEncoding(RecursiveToDeviceMixin):
    data: dict
    y: torch.FloatTensor
    batch_size: int

    def __len__(self):
        return self.batch_size


class ChembertaFeaturizer(PretrainedFeaturizerMixin[Tuple[dict, float], ChembertaBatchEncoding, ChembertaConfig]):
    def __init__(self, config: ChembertaConfig):
        super().__init__(config)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.pretrained_name)

    def _collate_encodings(self, encodings: List[Tuple[dict, float]]) -> ChembertaBatchEncoding:
        x_list, y_list = zip(*encodings)
        padded = self.tokenizer.pad(x_list, return_tensors='pt')
        data = {k: v for k, v in padded.items()}
        return ChembertaBatchEncoding(data=data,
                                      y=stack_y_list(y_list),
                                      batch_size=len(x_list))

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> Tuple[dict, float]:
        return self.tokenizer.encode_plus(smiles), y
