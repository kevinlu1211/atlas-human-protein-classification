# coding: utf-8

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")



import time
import pickle
from pathlib import Path
from functools import partial
import logging
from collections import defaultdict, Counter
from pprint import pprint

import click
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import torch
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from torchsummary import summary
from sklearn.metrics import f1_score
import scipy.optimize as opt

sys.path.append("../..")
from src.data import DataPaths, create_image_label_set, make_one_hot, open_rgby
from src.data import ProteinClassificationDataset, open_numpy, mean_proportion_class_weights, dataset_lookup, \
    sampler_weight_lookup, split_method_lookup, single_class_counter
from src.training import training_scheme_lookup
from src.models import model_lookup
from src.transforms import augment_fn_lookup
from src.callbacks import OutputRecorder

import pytorch_toolbox.fastai.fastai as fastai
from pytorch_toolbox.utils.core import to_numpy
from pytorch_toolbox.fastai_extensions.vision.utils import denormalize_fn_lookup, normalize_fn_lookup, tensor2img
from pytorch_toolbox.fastai.fastai.callbacks import CSVLogger
from pytorch_toolbox.fastai_extensions.basic_train import Learner
from pytorch_toolbox.fastai_extensions.loss import LossWrapper, loss_lookup
from pytorch_toolbox.fastai_extensions.basic_data import DataBunch
from pytorch_toolbox.fastai_extensions.callbacks import callback_lookup
from pytorch_toolbox.fastai_extensions.metrics import metric_lookup
from pytorch_toolbox.pipeline import PipelineGraph


def set_logger(log_level):
    log_levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NONSET": logging.NOTSET
    }
    logging.basicConfig(
        level=log_levels.get(log_level, logging.INFO),
    )


def load_training_data(root_image_paths, root_label_paths, use_n_samples=None):
    X = sorted(list(Path(root_image_paths).glob("*")), key=lambda p: p.stem)
    labels_df = pd.read_csv(root_label_paths)
    labels_df['Target'] = [[int(i) for i in s.split()] for s in labels_df['Target']]
    labels_df = labels_df.sort_values(["Id"], ascending=[True])
    if use_n_samples is not None:
        sampled_idx = np.random.choice(len(X), size=(use_n_samples)).flatten()
        X = np.array(X)[sampled_idx]
        labels_df = labels_df.iloc[sampled_idx]
    y = labels_df['Target'].values
    y_one_hot = make_one_hot(y, n_classes=28)
    assert np.all(np.array([p.stem for p in X]) == labels_df["Id"])
    return np.array(X), np.array(y), np.array(y_one_hot)


def load_testing_data(root_image_paths):
    X = sorted(list(Path(root_image_paths).glob("*")), key=lambda p: p.stem)
    return np.array(X)


def create_data_bunch(train_idx, val_idx, train_X, train_y, test_X, train_ds, train_bs, val_ds, val_bs, test_ds,
                      test_bs, sampler, num_workers):
    sampler = sampler(y=train_y[train_idx])
    train_ds = train_ds(inputs=train_X[train_idx], labels=train_y[train_idx])
    val_ds = val_ds(inputs=train_X[val_idx], labels=train_y[val_idx])
    test_ds = test_ds(inputs=test_X)
    return DataBunch.create(train_ds, val_ds, test_ds,
                            train_bs=train_bs, val_bs=val_bs, test_bs=test_bs,
                            collate_fn=train_ds.collate_fn, sampler=sampler, num_workers=num_workers)


def create_sampler(y=None, sampler_fn=None):
    sampler = None
    if sampler_fn is not None:
        weights = np.array(sampler_fn(y))
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
    else:
        pass
    return sampler


def create_callbacks(callback_references):
    callbacks = []
    for cb_ref in callback_references:
        callbacks.append(cb_ref())
    return callbacks


def create_learner_callbacks(learner_callback_references):
    callback_fns = []
    for learn_cb_ref in learner_callback_references:
        try:
            callback_fns.append(learn_cb_ref())
        except TypeError:
            callback_fns.append(learn_cb_ref)
    return callback_fns


def create_learner(data, model_creator, callbacks_creator, callback_fns_creator, metrics, loss_funcs):
    model = model_creator()
    callbacks = callbacks_creator()
    callback_fns = callback_fns_creator()
    learner = Learner(data,
                      model=model,
                      loss_func=LossWrapper(loss_funcs),
                      metrics=metrics,
                      callbacks=callbacks,
                      callback_fns=callback_fns)
    return learner


def training_loop(create_learner, data_bunch_creator, config_saver, data_splitter_iterable, training_scheme, record_results, state_dict):
    for i, (train_idx, val_idx) in enumerate(data_splitter_iterable(), 1):
        state_dict["current_fold"] = i
        config_saver()
        data = data_bunch_creator(train_idx, val_idx)
        learner = create_learner(data)
        training_scheme(learner)
        record_results(learner)


class ResultRecorder(fastai.Callback):
    _order = -0.5

    def __init__(self):
        self.names = []
        self.prob_preds = []
        self.targets = []

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if train:
            self.phase = 'TRAIN'
        else:
            label = last_target.get('label')
            if label is not None:
                self.phase = 'VAL'
            else:
                self.phase = 'TEST'
                #         inputs = tensor2img(last_input, denorm_fn=image_net_denormalize)
                #         self.inputs.extend(inputs)
        print([last_target['name'][0].split("_crop")[0]])
        self.names.extend([last_target['name'][0].split("_crop")[0]])
        if self.phase == 'TRAIN' or self.phase == 'VAL':
            label = to_numpy(last_target['label'])
            self.targets.extend(label)

    def on_loss_begin(self, last_output, **kwargs):
        prob_pred = to_numpy(torch.sigmoid(last_output))
        self.prob_preds.extend(prob_pred)

def save_config(save_path_creator, state_dict):
    save_path = save_path_creator()
    save_path.mkdir(parents=True, exist_ok=True)
    with (save_path / "config.yml").open('w') as yaml_file:
        yaml.dump(state_dict["config"], yaml_file, default_flow_style=False)

def record_results(learner, result_recorder_creator, save_path_creator):
    save_path = save_path_creator()

    # Save the optimal threshold result
    res_recorder = result_recorder_creator()
    learner.predict_on_dl(dl=learner.data.valid_dl, callbacks=[res_recorder])
    targets = np.stack(res_recorder.targets)
    pred_probs = np.stack(res_recorder.prob_preds)
    th = fit_val(pred_probs, targets)
    th[th < 0.1] = 0.1
    print('Thresholds: ', th)
    print('F1 macro: ', f1_score(targets, pred_probs > th, average='macro'))
    print('F1 macro (th = 0.5): ', f1_score(targets, pred_probs > 0.5, average='macro'))
    print('F1 micro: ', f1_score(targets, pred_probs > th, average='micro'))

    res_recorder = result_recorder_creator()
    learner.predict_on_dl(dl=learner.data.test_dl, callbacks=[res_recorder])
    names = np.stack(res_recorder.names)
    pred_probs = np.stack(res_recorder.pred_probs)
    predicted = []
    predicted_optimal = []
    for pred in pred_probs:
        classes = [str(c) for c in np.where(pred > 0.5)[0]]
        classes_optimal = [str(c) for c in np.where(pred > th)[0]]
        if len(classes) == 0:
            classes = [str(np.argmax(pred[0]))]
        if len(classes_optimal) == 0:
            classes_optimal = [str(np.argmax(pred[0]))]
        predicted.append(" ".join(classes))
        predicted_optimal.append(" ".join(classes_optimal))

    submission_df = pd.DataFrame({
        "Id": names,
        "Predicted": predicted
    })
    submission_df.to_csv(save_path / "submission.csv", index=False)
    optimal_submission_df = pd.DataFrame({
        "Id": names,
        "Predicted": predicted
    })
    optimal_submission_df.to_csv(save_path / "submission_optimal_threshold.csv", index=False)


def create_time_stamped_save_path(save_path, state_dict):
    current_time = state_dict.get("start_time")
    if current_time is None:
        current_time = f"{time.strftime('%Y%m%d-%H%M%S')}"
        state_dict["start_time"] = current_time
    current_fold = state_dict.get("current_fold")
    path = Path(save_path, current_time)
    if current_fold is not None:
        path = path / f"Fold_{current_fold}"
    return path


def create_output_recorder(save_path_creator, denormalize_fn):
    return partial(OutputRecorder, save_path=save_path_creator(),
                   save_img_fn=partial(tensor2img, denormalize_fn=denormalize_fn))


def create_csv_logger(save_path_creator):
    return partial(CSVLogger, filename=str(save_path_creator() / 'history'))


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def F1_soft(preds, targs, th=0.5, d=50.0):
    preds = sigmoid_np(d * (preds - th))
    targs = targs.astype(np.float)
    score = 2.0 * (preds * targs).sum(axis=0) / ((preds + targs).sum(axis=0) + 1e-6)
    return score


def fit_val(x, y):
    params = 0.5 * np.ones(28)
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x, y, p) - 1.0,
                                      wd * (p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p


learner_callback_lookup = {
    "create_output_recorder": create_output_recorder,
    "create_csv_logger": create_csv_logger,
    "GradientClipping": fastai.GradientClipping,
}

callback_lookup = {
    "ResultRecorder": ResultRecorder,
    **callback_lookup,
}

lookups = {
    **model_lookup,
    **dataset_lookup,
    **sampler_weight_lookup,
    **split_method_lookup,
    **augment_fn_lookup,
    **normalize_fn_lookup,
    **denormalize_fn_lookup,
    **loss_lookup,
    **metric_lookup,
    **callback_lookup,
    **sampler_weight_lookup,
    **learner_callback_lookup,
    **training_scheme_lookup,
    "open_numpy": open_numpy,
    "load_training_data": load_training_data,
    "load_testing_data": load_testing_data,
    "create_data_bunch": create_data_bunch,
    "create_sampler": create_sampler,
    "create_learner": create_learner,
    "create_time_stamped_save_path": create_time_stamped_save_path,
    "create_callbacks": create_callbacks,
    'create_learner_callbacks': create_learner_callbacks,
    "training_loop": training_loop,
    "record_results": record_results,
    "save_config": save_config
}


@click.command()
@click.option('-cfg', '--config_file_path')
@click.option('-log-lvl', '--log_level', default="INFO")
def main(config_file_path, log_level):
    set_logger(log_level)
    with Path(config_file_path).open("r") as f:
        config = yaml.load(f)
    pipeline_graph = PipelineGraph.create_pipeline_graph_from_config(config)
    print(pipeline_graph.sorted_node_names)
    pipeline_graph.run_graph(reference_lookup=lookups)


if __name__ == '__main__':
    main()
