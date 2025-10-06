from typing import Iterable, Optional

from torch.utils.data import Dataset

from utils.data_manager1 import IncrementalDataManager


class DataManager:
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
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

    def _prepare_subset(self, subset: Dataset, mode: Optional[str]) -> Dataset:
        if mode is not None:
            transform = self._idm._build_transform(mode)
            setattr(subset, "transform", transform)
        return subset

    def get_subset(self, task: int, source: str, cumulative: bool = False, mode: Optional[str] = None) -> Dataset:
        transform = self._idm._build_transform(mode)
        subset = self._idm.get_subset(task, source=source, cumulative=cumulative, transform=transform)
        return subset
