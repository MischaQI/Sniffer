"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.newsclip_newsvqa_datasets import (
    NewsClipNewsVQADataset,
    NewsClipNewsVQAEvalDataset
)

from lavis.common.registry import registry

import os
import logging
import warnings

import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process



@registry.register_builder("newsclip_newsvqa")
class NewsClipNewsVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = NewsClipNewsVQADataset
    eval_dataset_cls = NewsClipNewsVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/newsclip/defaults_newsvqa.yaml",
    }

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = os.path.join(registry.get_path("repo_root"),"datasets",ann_path)
                    # ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                # vis_path = utils.get_cache_path(vis_path)
                vis_path = os.path.join(registry.get_path("repo_root"),"datasets",vis_path)

            if not os.path.exists(vis_path):
                # warnings.warn("storage path {} does not exist.".format(vis_path))
                raise RuntimeError("storage path {} does not exist.".format(vis_path))


            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
            )

        return datasets
