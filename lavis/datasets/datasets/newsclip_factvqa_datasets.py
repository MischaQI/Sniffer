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
                "caption": ann["caption"],
                "answer": ann["answer"],
                "image": sample["img_path"],
            }
        )

class NewsClipFactVQADataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, visentity_root, use_visentity):
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

        self.use_visentity = use_visentity

        if self.use_visentity:
            self.visentities = json.load(open(visentity_root))
        
        
    def get_instruction(self, caption):
        prompt = "Some rumormongers use images from other events as illustrations of current news event to make multimodal misinformation. Given a news caption and a news image, judge whether the given image is wrongly used in a different news context. Let's analyze their inconsistency from perspectives of main news elements, including time, place, person, event, artwork, etc. You should answer in the following forms: 'No, the image is rightly used.' or 'Yes, the image is wrongly used in a different news context. The given news caption and image are inconsistent in <element>. The <element> in caption is <entity_1>, and the <element> in image is <entity_2>. ' The news caption is '{}'. The answer is ".format(caption)

        return prompt
    
    def get_instruction_visentity(self, caption, dict_visentity):
        entities_cut = dict_visentity['str_visent_cut'] 
        prompt = f"Some rumormongers use images from other events as illustrations of current news event to make multimodal misinformation. Given a news caption and a news image, judge whether the given image is wrongly used in a different news context. Let's analyze their inconsistency from perspectives of main news elements, including time, place, person, event, artwork, etc. You should answer in the following forms: 'No, the image is rightly used.' or 'Yes, the image is wrongly used in a different news context. The given news caption and image are inconsistent in <element>. The <element> in caption is <entity_1>, and the <element> in image is <entity_2>. ' The news caption is '{caption}'. The possible visual entities is {','.join(entities_cut)}. The answer is "

        return prompt
    
    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["img_path"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        if self.use_visentity == False:
            caption = self.text_processor( self.get_instruction( ann["caption"] ) )
        else:
            if ann["idx"] in self.visentities.keys():
                dict_visentity = self.visentities[ann["idx"]]
                caption = self.text_processor( self.get_instruction_visentity( ann["caption"], dict_visentity) )
            else:
                caption = self.text_processor( self.get_instruction( ann["caption"] ) )

            
            

        answer = self.text_processor(ann['answer'])

        return {
            "image": image,
            "text_input": caption,
            "text_output": answer,
            "image_id": self.img_ids[ann["idx"]],
        }


class NewsClipFactVQAEvalDataset(NewsClipFactVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, visentity_root, use_visentity):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths,visentity_root, use_visentity)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["img_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        if self.use_visentity == False:
            caption = self.text_processor( self.get_instruction( ann["caption"] ) )
        else:
            if ann["idx"] in self.visentities.keys():
                dict_visentity = self.visentities[ann["idx"]]
                caption = self.text_processor( self.get_instruction_visentity( ann["caption"], dict_visentity) )
            else:
                caption = self.text_processor( self.get_instruction( ann["caption"] ) )


        try:
            answer = self.text_processor(ann['answer'])
        except:
            answer = self.text_processor("")

        return {
            "image": image,
            "text_input": caption,
            "text_output": answer,
            "image_id": ann["idx"],
            "instance_id": ann["instance_id"],
        }

