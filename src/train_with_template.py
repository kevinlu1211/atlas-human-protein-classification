# coding: utf-8

import sys
import warnings
import pickle

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import time
from functools import partial
import logging
from typing import List, Union, Tuple
from pathlib import Path
from collections import Counter
import click
import yaml
import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import f1_score
import scipy.optimize as opt

sys.path.append("../..")

from src.data import make_one_hot, open_numpy, dataset_lookup, \
    sampler_weight_lookup, split_method_lookup, Image
from src.training import training_scheme_lookup
from src.models import model_lookup
from src.transforms import augment_fn_lookup
from src.callbacks import OutputRecorder, ResultRecorder

import pytorch_toolbox.fastai.fastai as fastai
from pytorch_toolbox.utils import listify
from pytorch_toolbox.fastai_extensions.vision.utils import denormalize_fn_lookup, normalize_fn_lookup, tensor2img
from pytorch_toolbox.fastai.fastai.callbacks import CSVLogger
from pytorch_toolbox.fastai_extensions.basic_train import Learner, DataBunch
from pytorch_toolbox.fastai_extensions.loss import LossWrapper, loss_lookup
from pytorch_toolbox.fastai_extensions.callbacks import callback_lookup, learner_callback_lookup
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


@click.command()
@click.option('-cfg', '--config_file_path')
@click.option('-log-lvl', '--log_level', default="INFO")
def main(config_file_path, log_level):
    set_logger(log_level)
    with Path(config_file_path).open("r") as f:
        config = yaml.load(f)
    pipeline_graph = PipelineGraph.create_pipeline_graph_from_config(config)
    pipeline_graph.run(reference_lookup=lookups)
    # pipeline_graph.run(reference_lookup=lookups, to_node="CreateInference")
    # create_inference_fn = pipeline_graph.get_node_output("CreateInference")
    # image = get_image_from_class(('Plasma membrane', 'Cell junctions'))[0]
    # names, prediction_probs = create_inference_fn(image)


def load_training_labels(training_labels_path):
    labels_df = pd.read_csv(training_labels_path)
    labels_df['Target'] = [[int(i) for i in s.split()] for s in labels_df['Target']]
    return labels_df


def create_image_id_lookups(labels_df):
    image_id_with_labels = labels_df['Id']
    image_id_with_labels_lookup = Counter(image_id_with_labels)
    return image_id_with_labels_lookup


def load_training_images(training_images_path):
    image_paths = []
    for p in listify(training_images_path):
        image_paths.extend(Path(p).glob("*"))
    return image_paths


def filter_image_paths_with_labels(image_paths, labels_df):
    # We use a Counter to filter in O(n) instead of O(n^2) time
    image_id_with_labels_lookup = create_image_id_lookups(labels_df)
    image_paths_used_for_training = [Path(p) for p in image_paths if
                                     image_id_with_labels_lookup.get(Path(p).stem) is not None]
    return image_paths_used_for_training


def load_training_data(root_image_paths: Union[List[str], str], root_label_paths: List[str],
                       use_n_samples: Union[None, int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_df = load_training_labels(root_label_paths)
    image_paths = load_training_images(root_image_paths)
    image_paths_with_labels = filter_image_paths_with_labels(image_paths, labels_df)

    sorted_labels_df = labels_df.sort_values(["Id"], ascending=[True])
    sorted_image_paths_used_for_training = sorted(image_paths_with_labels, key=lambda x: x.stem)

    if len(sorted_image_paths_used_for_training) != len(sorted_labels_df["Id"]):
        logging.warning("Number of images don't match up with number of labels")
    else:
        if not (np.all(np.array([p.stem for p in sorted_image_paths_used_for_training]) == sorted_labels_df["Id"])):
            logging.warning("Images and training labels aren't matched up!")

    if use_n_samples is not None:
        sampled_idx = np.random.choice(len(sorted_image_paths_used_for_training), size=use_n_samples).flatten()
        sorted_image_paths_used_for_training = np.array(sorted_image_paths_used_for_training)[sampled_idx]
        sorted_labels_df = sorted_labels_df.iloc[sampled_idx]

    labels = sorted_labels_df['Target'].values
    labels_one_hot = make_one_hot(labels, n_classes=28)
    return np.array(sorted_image_paths_used_for_training), np.array(labels), np.array(labels_one_hot)


def load_testing_data(root_image_paths, use_n_samples=None):
    X = sorted(list(Path(root_image_paths).glob("*")), key=lambda p: p.stem)
    if use_n_samples:
        X = X[:use_n_samples]
    return np.array(X)


def create_data_bunch(train_idx, val_idx, train_X, train_y_one_hot, train_y, test_X, train_ds, train_bs, val_ds, val_bs,
                      test_ds,
                      test_bs, sampler, num_workers):
    sampler = sampler(y=train_y[train_idx])
    train_ds = train_ds(inputs=train_X[train_idx], labels=train_y_one_hot[train_idx])
    val_ds = val_ds(inputs=train_X[val_idx], labels=train_y_one_hot[val_idx])
    test_ds = test_ds(inputs=test_X)
    return DataBunch.create(train_ds, val_ds, test_ds,
                            train_bs=train_bs, val_bs=val_bs, test_bs=test_bs,
                            collate_fn=train_ds.collate_fn, sampler=sampler, num_workers=num_workers)


def create_data_bunch_for_inference(X_test, ds, num_workers):
    train_ds = ds(inputs=X_test, image_cached=True)
    val_ds = ds(inputs=X_test, image_cached=True)
    test_ds = ds(inputs=X_test, image_cached=True)
    return DataBunch.create(train_ds, val_ds, test_ds, num_workers=num_workers, collate_fn=test_ds.collate_fn)


def create_sampler(y=None, sampler_fn=None):
    sampler = None
    if sampler_fn is not None:
        weights = np.array(sampler_fn(y))

        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
    else:
        pass

    if sampler is not None:
        label_cnt = Counter()
        n_samples = len(weights)
        for idx in np.random.choice(len(y), n_samples, p=weights / weights.sum()):
            labels = y[idx]
            for l in labels:
                label_cnt[l] += 1
        # print("Weighted sampled proportions:")
        # pprint(sorted({k: v / sum(label_cnt.values()) for k, v in label_cnt.items()}.items()))
        # pprint(sorted({k: v for k, v in name_cnt.items()}.items(), key=lambda x: x[1]))
    # else:
    #     print("No weighted sampling")
    return sampler


def create_callbacks(callback_references):
    callbacks = []
    for cb_ref in callback_references:
        try:
            callbacks.append(cb_ref())
        except TypeError:
            callbacks.append(cb_ref)
    return callbacks


def create_learner_callbacks(learner_callback_references):
    callback_fns = []
    for learn_cb_ref in learner_callback_references:
        try:
            callback_fns.append(learn_cb_ref())
        except TypeError:
            callback_fns.append(learn_cb_ref)
    return callback_fns


def create_learner(data, model_creator, loss_funcs=[], metrics=None,
                   callbacks_creator=None, callback_fns_creator=None, to_fp16=False, model_path=None):
    model = model_creator()
    callbacks = callbacks_creator() if callbacks_creator is not None else None
    callback_fns = callback_fns_creator() if callback_fns_creator is not None else None

    learner = Learner.create(data=data,
                             model=model,
                             layer_groups=get_layer_groups(model),
                             loss_func=LossWrapper(loss_funcs),
                             metrics=metrics,
                             callbacks=callbacks,
                             callback_fns=callback_fns)
    if model_path is not None:
        learner.load_from_path(model_path)
    if to_fp16:
        learner = learner.to_fp16()
    return learner


def get_layer_groups(model):
    try:
        return model.layer_groups
    except AttributeError:
        return None


def training_loop(create_learner, data_bunch_creator, config_saver, data_splitter_iterable, training_scheme,
                  record_results, state_dict):
    for i, (train_idx, val_idx) in enumerate(data_splitter_iterable(), 1):
        state_dict["current_fold"] = i
        config_saver()
        data = data_bunch_creator(train_idx, val_idx)
        learner = create_learner(data)
        training_scheme(learner)
        record_results(learner)


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
    pickle.dump(th, open(save_path / "thresholds.csv", "wb"))
    th[th < 0.1] = 0.1
    print('Thresholds: ', th)
    print('F1 macro: ', f1_score(targets, pred_probs > th, average='macro'))
    print('F1 macro (th = 0.5): ', f1_score(targets, pred_probs > 0.5, average='macro'))
    print('F1 micro: ', f1_score(targets, pred_probs > th, average='micro'))

    res_recorder = result_recorder_creator()
    learner.predict_on_dl(dl=learner.data.test_dl, callbacks=[res_recorder])
    names = np.stack(res_recorder.names)
    pred_probs = np.stack(res_recorder.prob_preds)
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
        "Predicted": predicted_optimal
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
    logging.info(f"Root path of experiment is: {path.parent}")
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


def create_inference(image, inference_data_bunch_creator, inference_learner_creator, result_recorder_creator):
    inference_data_bunch = inference_data_bunch_creator([Image(image)])
    inference_learner = inference_learner_creator(inference_data_bunch)
    result_recorder = result_recorder_creator()
    inference_learner.predict_on_dl(dl=inference_learner.data.test_dl, callbacks=[result_recorder])
    return np.stack(result_recorder.names), np.stack(result_recorder.prob_preds)


learner_callback_lookup = {
    "create_output_recorder": create_output_recorder,
    "create_csv_logger": create_csv_logger,
    "GradientClipping": fastai.GradientClipping,
    **learner_callback_lookup
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
    "create_data_bunch_for_inference": create_data_bunch_for_inference,
    "create_sampler": create_sampler,
    "create_learner": create_learner,
    "create_time_stamped_save_path": create_time_stamped_save_path,
    "create_callbacks": create_callbacks,
    'create_learner_callbacks': create_learner_callbacks,
    "training_loop": training_loop,
    "record_results": record_results,
    "save_config": save_config,
    "create_inference": create_inference
}

if __name__ == '__main__':
    main()
