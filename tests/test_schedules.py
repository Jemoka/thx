from dataclasses import dataclass
from types import SimpleNamespace

import jax
import numpy as np
import optax
import pytest

from theseus.config import build, configuration, field
from theseus.model.module import Module
from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.training.schedules import (
    CosineRewarm,
    Schedule,
    WSD,
    WSDS,
)


class ScheduleModel(Module):
    @classmethod
    def components(cls):
        return []


class ScheduleTrainer(BaseTrainer):
    MODEL = ScheduleModel
    CONFIG = BaseTrainerConfig
    DATASET = []
    SCHEDULE = WSD


@pytest.mark.parametrize(
    "schedule, config, steps, expected",
    [
        (
            WSD,
            dict(lr=1.0, warmup_pct=0.1, decay_pct=0.2),
            [0, 10, 80, 100],
            [0.01, 1, 1, 0.01],
        ),
        (
            WSDS,
            dict(lr=1.0, warmup_pct=0.1, decay_pct=0.2, constant_pct=0.3),
            [0, 10, 50, 70, 100],
            [0.01, 1, 1, 0.1, 0.1],
        ),
        (
            CosineRewarm,
            dict(
                lr=1.0,
                rewarm_lr=0.5,
                min_lr_mult=0.01,
                warmup_pct=0.1,
                stage_tokens=[100, 100],
                batch_size=1,
                block_size=1,
            ),
            [0, 10, 100, 110, 200],
            [0.01, 1, 0.01, 0.5, 0.005],
        ),
    ],
)
def test_schedule_phase_values(schedule, config, steps, expected):
    lr = schedule.schedule(steps[-1], schedule.config(**config))
    np.testing.assert_allclose(
        [jax.jit(lr)(step) for step in steps], expected, atol=1e-7
    )


@pytest.mark.parametrize("schedule", [WSD, WSDS, CosineRewarm])
def test_trainer_collects_and_builds_schedule(schedule):
    class Trainer(ScheduleTrainer):
        SCHEDULE = schedule

    assert schedule.config in Trainer.config()
    cfg = build(*Trainer.config())
    cfg.optimization.lr = 0.125
    trainer = object.__new__(Trainer)
    trainer.total_steps = 1000
    trainer.args = SimpleNamespace(lr=0.125)
    with configuration(cfg):
        lr = trainer.schedule
        assert trainer.schedule is lr
    assert float(lr(0)) == pytest.approx(0.00125)


def test_custom_schedule_and_constant_override():
    @dataclass
    class CustomConfig:
        lr: float = field("optimization/lr", default=0.25)
        divisor: int = field("optimization/divisor", default=2)

    def custom(total_steps, cfg):
        return optax.linear_schedule(cfg.lr, cfg.lr / cfg.divisor, total_steps)

    class CustomTrainer(ScheduleTrainer):
        SCHEDULE = Schedule(CustomConfig, custom)

    class ConstantTrainer(CustomTrainer):
        SCHEDULE = None

    assert CustomConfig in CustomTrainer.config()
    assert CustomConfig not in ConstantTrainer.config()
    cfg = build(*CustomTrainer.config())
    cfg.optimization.lr = 0.5
    cfg.optimization.divisor = 4
    trainer = object.__new__(CustomTrainer)
    trainer.total_steps = 100
    with configuration(cfg):
        lr = trainer.schedule
        assert trainer.schedule is lr
    assert float(lr(0)) == pytest.approx(0.5)
    assert float(lr(100)) == pytest.approx(0.125)

    constant = object.__new__(ConstantTrainer)
    constant.args = SimpleNamespace(lr=0.75)
    lr = constant.schedule
    assert lr(0) == lr(1000) == 0.75
