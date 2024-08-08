"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import json


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["img_path"],
                "question": ann["question"],
                "answer": ann["answer"],
                "image": sample["img_path"],
            }
        )

class NewsClipNewsVQADataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["idx"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["img_path"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
                
        question = self.text_processor(ann["question"])
        answer = self.text_processor(ann['answer'])

        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
            "image_id": self.img_ids[ann["idx"]],
        }


class NewsClipNewsVQAEvalDataset(NewsClipNewsVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["img_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        question = self.text_processor(ann['question'])

        try:
            answer = self.text_processor(ann['answer'])
        except:
            answer = self.text_processor("")

        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
            "image_id": ann["idx"],
            "instance_id": ann["instance_id"],
        }

