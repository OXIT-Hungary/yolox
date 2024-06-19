#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.num_classes = 3

        self.data_dir = "/home/geri/work/datasets/wp_dataset/"
        self.train_ann = "train_annotations.coco.json"
        self.val_ann = "val_annotations.coco.json"
        self.test_ann = "test_annotations.coco.json"

        self.output_dir = "/home/geri/work/OXIT-Sport_Framework/src/yolox/output"

        self.max_epoch = 150
