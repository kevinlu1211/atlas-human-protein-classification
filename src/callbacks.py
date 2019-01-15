from collections import defaultdict

import torch
import numpy as np
import pandas as pd

from pytorch_toolbox.utils import to_numpy
import pytorch_toolbox.fastai.fastai as fastai


class ResultRecorder(fastai.Callback):
    _order = -10

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
        self.names.extend(last_target['name'])
        if self.phase == 'TRAIN' or self.phase == 'VAL':
            label = to_numpy(last_target['label'])
            self.targets.extend(label)

    def on_loss_begin(self, last_output, **kwargs):
        prob_pred = to_numpy(torch.sigmoid(last_output))
        self.prob_preds.extend(prob_pred)


class OutputRecorder(fastai.LearnerCallback):
    _order = -10

    def __init__(self, learn, save_path, save_img_fn, save_img=False):
        super().__init__(learn)
        self.save_path = save_path
        self.history = defaultdict(list)
        self.phase = None
        self.current_batch = dict()
        self.save_img_fn = save_img_fn
        self.save_img = save_img

    def on_batch_begin(self, last_input, last_target, epoch, train, **kwargs):
        if train:
            self.phase = 'TRAIN'
        else:
            label = last_target.get('label')
            if label is not None:
                self.phase = 'VAL'
            else:
                self.phase = 'TEST'
        self.key = (self.phase, epoch)
        if self.save_img:
            inputs = self.save_img_fn(last_input)
            self.current_batch['input'] = inputs
        self.current_batch['name'] = last_target['name']
        if self.phase == 'TRAIN' or self.phase == 'VAL':
            label = to_numpy(last_target['label'])
            self.current_batch['label'] = label

    def on_loss_begin(self, last_output, epoch, **kwargs):
        model_output = to_numpy(last_output)
        prediction_probs = 1 / (1 + np.exp(-model_output))
        self.current_batch['prediction_probs'] = prediction_probs
        prediction = prediction_probs.copy()
        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 1
        self.current_batch['prediction'] = prediction

    def on_batch_end(self, **kwargs):
        for loss in self.learn.loss_func.losses:
            name = loss.__class__.__name__
            unreduced_loss = to_numpy(loss.loss)
            # reduced_loss = to_numpy(loss.loss.mean())
            self.current_batch[f"{name}"] = unreduced_loss
            # self.current_batch[f"{name}_reduced"] = reduced_loss
        # prediction = self.current_batch['prediction']
        # label = self.current_batch['label']
        # n_classes = label.shape[-1]
        # indices_to_keep = np.where((prediction == label).sum(axis=1) != n_classes)[0]
        if self.phase == 'TRAIN' or self.phase == 'VAL':

            """
            self.current_batch is a dictionary with:
            {
                stat1: [stat1_for_sample_1, stat1_for_sample_2, ...]
            }
            """

            # Get the keys, and the array of values associated with each key
            stat_names, stat_values = zip(*self.current_batch.items())

            # zip the array of values so each element in the zip has [stat1_for_sample_1, stat2_for_sample_1, ...]
            stat_values_for_samples = zip(*stat_values)

            for stat_values_for_sample in stat_values_for_samples:
                sample_to_save = dict()
                for stat_name, stat_value in zip(stat_names, stat_values_for_sample):
                    if stat_name == 'input': continue
                    sample_to_save[stat_name] = stat_value
                self.history[self.key].append(sample_to_save)

    def on_epoch_end(self, epoch, **kwargs):
        prev_epoch = epoch - 1
        n_cycle = self.learn.n_cycle
        history_save_path = self.save_path / 'training_logs' / f"cycle_{n_cycle}_epoch_{prev_epoch}_train.csv"
        history_save_path.parent.mkdir(exist_ok=True, parents=True)
        history = self.history[('TRAIN', prev_epoch)]
        df = pd.DataFrame(history)
        df.to_csv(history_save_path, index=False)

        history_save_path = self.save_path / 'training_logs' / f"cycle_{n_cycle}_epoch_{prev_epoch}_val.csv"
        history_save_path.parent.mkdir(exist_ok=True, parents=True)
        history = self.history[('VAL', prev_epoch)]
        df = pd.DataFrame(history)
        df.to_csv(history_save_path, index=False)

        model_save_path = self.save_path / 'model_checkpoints' / f"cycle_{n_cycle}_epoch_{prev_epoch}"
        model_save_path.parent.mkdir(exist_ok=True, parents=True)
        self.learn.save(model_save_path)
        self.history = defaultdict(list)


import numpy as np
import cv2
import os
import glob

channelColors = ['red', 'green', 'blue', 'yellow']


def readChannels(root_dir, imgid):
    channels = []
    for color in channelColors:
        imagePath = root_dir + '/' + imgid + '_' + color + '.tif'
        chan = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        channels.append(chan)
    channels = np.array(channels)
    return channels


def getImageIds(root_dir):
    imageFilepaths = glob.glob(root_dir + '/*.png')
    imgids = []
    for fp in imageFilepaths:
        d, f = os.path.split(fp)
        name, ext = os.path.splitext(f)
        fid, color = name.split('_')
        imgids.append(fid)
    imgids = list(set(imgids))
    return imgids


def makeImagePath(root_dir, imgid):
    path = root_dir + '/' + imgid + '.npy'
    return path


def makeComposites(root_dir, save_dir, force=False):
    imgids = getImageIds(root_dir)
    for imgid in imgids:
        imgPath = makeImagePath(root_dir, imgid)
        if force or not os.path.exists(imgPath):
            channels = readChannels(root_dir, imgid)
            np.save(imgPath, channels, allow_pickle=True)


def readComposite(root_dir, imgid):
    imgPath = makeImagePath(root_dir, imgid)
    channels = np.load(imgPath, allow_pickle=True)
    return channels
