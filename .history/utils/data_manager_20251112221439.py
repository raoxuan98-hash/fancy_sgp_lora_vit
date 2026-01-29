from typing import Iterable, Optional

from torch.utils.data import Dataset

from utils.data_manager1 import IncrementalDataManager
from utils.cross_domain_data_manager import CrossDomainDataManagerCore


from typing import Optional, List
from torch.utils.data import Dataset

class WithinDomainDataManager:
    def __init__(self, dataset_name: str, shuffle: bool, seed: int, init_cls: int, increment: int, args: dict = None):
        self.args = args or {}
        self._idm = IncrementalDataManager(
            dataset_name=dataset_name,
            initial_classes=init_cls,
            increment_classes=increment,
            shuffle=shuffle,
            seed=seed)
        self.dataset_name = dataset_name

    @property
    def nb_tasks(self) -> int:
        return self._idm.nb_tasks

    def get_task_size(self, task: int) -> int:
        return self._idm.get_task_size(task)

    def get_task_classes(self, task_id: int, cumulative: bool = False):
        return self._idm.get_task_classes(task_id, cumulative=cumulative)

    def _prepare_subset(self, subset: Dataset, mode: Optional[str]) -> Dataset:
        if mode is not None:
            transform = self._idm._build_transform(mode)
            setattr(subset, "transform", transform)
        return subset

    def get_subset(self, task: int, source: str, cumulative: bool = False, mode: Optional[str] = None) -> Dataset:
        transform = self._idm._build_transform(mode) if mode is not None else None
        subset = self._idm.get_subset(task, source=source, cumulative=cumulative, transform=transform)
        return self._prepare_subset(subset, mode)


class CrossDomainDataManager:
    def __init__(self, dataset_name: str, shuffle: bool, seed: int, args: dict = None):
        self.args = args or {}
        dataset_names = self.args.get('cross_domain_datasets', [
            'caltech-101', 'dtd', 'eurosat_clip', 'fgvc-aircraft-2013b-variants102',
            'food-101', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets',
            'stanford-cars', 'imagenet-r'])
        
        self._cdm = CrossDomainDataManagerCore(
            dataset_names=dataset_names,
            shuffle=shuffle,
            seed=seed)
        self.dataset_name = dataset_name

    @property
    def nb_tasks(self) -> int:
        return self._cdm.nb_tasks

    def get_task_size(self, task: int) -> int:
        return self._cdm.get_task_size(task)

    def get_task_classes(self, task_id: int, cumulative: bool = False):
        return self._cdm.get_task_classes(task_id, cumulative=cumulative)

    def _prepare_subset(self, subset: Dataset, mode: Optional[str]) -> Dataset:
        if mode is not None:
            transform = self._cdm._build_transform(mode)
            setattr(subset, "transform", transform)
        return subset

    def get_subset(self, task: int, source: str, cumulative: bool = False, mode: Optional[str] = None) -> Dataset:
        transform = self._cdm._build_transform(mode) if mode is not None else None
        subset = self._cdm.get_subset(task, source=source, cumulative=cumulative, transform=transform)
        return self._prepare_subset(subset, mode)