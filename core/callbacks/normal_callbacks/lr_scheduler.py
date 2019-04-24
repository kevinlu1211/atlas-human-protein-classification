from itertools import cycle
from dataclasses import dataclass

import numpy as np

from pytorch_toolbox.core.defaults import Collection, Any, Number, List, Floats, Optional
from pytorch_toolbox.core.callbacks.core import Callback, Scheduler, annealing_no
from pytorch_toolbox.core.utils import if_none, listify


@dataclass
class TrainingPhase:
    "Schedule hyper-parameters for a phase of `length` iterations."
    length: int

    def __post_init__(self): self.scheduler_lookup = dict()

    def schedule_hp(self, name, vals, anneal=None):
        "Adds a schedule for `name` between `vals` using `anneal`."
        self.scheduler_lookup[name] = Scheduler(vals, n_iter=self.length, func=anneal)
        return self

    @property
    def is_done(self):
        return list(self.scheduler_lookup.values())[0].is_done

    @property
    def hyperparameter_and_scheduler_pairs(self):
        return self.scheduler_lookup.items()


class GeneralScheduler(Callback):
    "Schedule multiple `TrainingPhase` for a `Learner`."

    def __init__(self, learn, phases: Collection[TrainingPhase], start_epoch: int = None):
        super().__init__()
        self.learn = learn
        self.phases, self.start_epoch = phases, start_epoch
        self.current_phase_idx = 0

    @property
    def current_phase(self):
        return self.phases[self.current_phase_idx]

    def on_train_begin(self, epoch: int, **kwargs: Any) -> None:
        "Initialize the schedulers for training."
        res = {'epoch': self.start_epoch} if self.start_epoch is not None else None
        self.start_epoch = if_none(self.start_epoch, epoch)
        self.opt = self.learn.opt
        current_hp_name_and_scheduler_pairs = self.current_phase.hyperparameter_and_scheduler_pairs
        for hp_name, scheduler in current_hp_name_and_scheduler_pairs:
            scheduler.restart()
            setattr(self.opt, hp_name, scheduler.start)
        return res

    def jump_to_epoch(self, epoch: int) -> None:
        for _ in range(len(self.learn.data.train_dl) * epoch):
            self.on_batch_end(True)

    def on_batch_end(self, train, **kwargs: Any) -> None:
        "Take a step in lr,mom sched, start next stepper when the current one is complete."
        if train:
            if self.current_phase_idx >= len(self.phases):
                return {'stop_training': True, 'stop_epoch': True}

            current_hp_name_and_scheduler_pairs = self.current_phase.hyperparameter_and_scheduler_pairs
            for hp_name, scheduler in current_hp_name_and_scheduler_pairs:
                hp_value = scheduler.step()
                setattr(self.opt, hp_name, hp_value)

            if self.current_phase.is_done:
                self.current_phase_idx += 1


class MultiStepScheduler(GeneralScheduler):

    def __init__(self, learn, epochs_for_step: List[Number], hyperparameter_name: str,
                 hyperparameter_values_for_step: List[Floats],
                 start_epoch: Optional[Number] = None,
                 end_epoch: Optional[Number] = None):
        assert len(epochs_for_step) > 0
        assert epochs_for_step[0] == 0, "Please set the first epoch to 0"
        if len(epochs_for_step) == 1:
            assert end_epoch is not None, "Please provide an end epoch as it can't be automatically calculated"

        self.end_epoch = if_none(end_epoch, epochs_for_step[-1] * 2)
        self.epochs_for_step = epochs_for_step
        self.hyperparameter_values_for_step = hyperparameter_values_for_step
        self.hyperparameter_name = hyperparameter_name
        self.epochs_to_iterations = self._convert_epochs_to_iterations(n_iterations_in_epoch=len(learn.data.train_dl))
        self.training_phases = self._create_training_phases()
        super().__init__(learn, self.training_phases, start_epoch)

    def _convert_epochs_to_iterations(self, n_iterations_in_epoch):
        epochs_between_each_step = self._calculate_epochs_between_each_step()
        epochs_to_n_iterations = [int(epoch * n_iterations_in_epoch) for epoch in epochs_between_each_step]
        return epochs_to_n_iterations

    def _calculate_epochs_between_each_step(self):
        """
        Calculates the number of epochs between each step

        For len(self.epochs_for_step) == 1:
        let self.epochs_for_step = [0]
        let self.end_epoch = 80
        epochs_for_step_with_end_epoch = [0, 80]
        epochs_between_each_step = [80] - [0] = [80]

        For len(self.epochs_for_step) > 1:
        let self.epochs_for_step = [0, 10, 30, 50]
        let self.end_epoch = 80
        epochs_for_step_with_end_epoch = [0, 10, 30, 50, 80]
        epochs_between_each_step = [10, 30, 50, 80] - [0, 10, 30, 50] = [10, 20, 20, 30]
        :return:
        """
        epochs_for_step_with_end_epoch = np.array(self.epochs_for_step + [self.end_epoch])
        epochs_between_each_step = epochs_for_step_with_end_epoch[1:] - np.array(self.epochs_for_step)
        return epochs_between_each_step

    def _create_training_phases(self):
        training_phases = []
        for hp_name, hp_val, n_iterations in zip(cycle([self.hyperparameter_name]),
                                                 self.hyperparameter_values_for_step,
                                                 self.epochs_to_iterations):
            training_phase = TrainingPhase(length=n_iterations)
            training_phase = training_phase.schedule_hp(hp_name, hp_val, anneal=annealing_no)
            training_phases.append(training_phase)
        return training_phases
