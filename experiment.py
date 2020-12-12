import datetime
from pathlib import Path

import torch


class Experiment:
    def __init__(self, name, model, hp):
        self._check_hp(hp)

        self.name = name
        self.model = model
        self.hp = hp

        self.step = 0
        self.episode = 0

    @classmethod
    def _check_hp(cls, hp):
        if type(hp) != dict:
            raise ValueError("`hp` must be a dictionary.")

    @classmethod
    def _create_name(cls, base_name: str):
        now = datetime.datetime.now()

        return (
            base_name
            + f"__{now.year}{now.month:02}{now.day:02}_{now.hour:02}{now.minute:02}"
        )

    @classmethod
    def _get_save_path(cls, name: str, base_path="saves"):
        return Path(f"{base_path}/{name}.pickle")

    @classmethod
    def _get_writer_path(cls, name: str, base_path="runs"):
        return Path(f"{base_path}/{name}")

    @classmethod
    def _get_log_path(cls, name: str, base_path="logs"):
        return Path(f"{base_path}/{name}.log")

    @classmethod
    def create(cls, base_name, model_class, hp):
        cls._check_hp(hp)

        model = model_class(**hp)
        return cls(cls._create_name(base_name), model=model, hp=hp)

    @classmethod
    def load(cls, name):
        save_path = cls._get_save_path(name)

        if not Path(save_path).is_file():
            raise ValueError(f"Could not find the save file `{save_path}`")

        with open(save_path, "rb") as file:
            return torch.load(file)

    @property
    def save_path(self):
        return self._get_save_path(self.name)

    @property
    def writer_path(self):
        return self._get_writer_path(self.name)

    @property
    def log_path(self):
        return self._get_log_path(self.name)

    def save(self):
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir()

        with open(self.save_path, "wb") as file:
            torch.save(self, file)

    def info(self, logger):
        if self.episode == 0:
            logger.info(f"Starting training at episode {self.episode}.")
        else:
            logger.info(f"Resuming training at episode {self.episode}.")

        for key in self.hp:
            logger.info(f"{key}: {self.hp[key]}")
