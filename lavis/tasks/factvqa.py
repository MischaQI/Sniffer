"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from lavis.common.dist_utils import main_process, is_main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.tasks.captioning import CaptionTask
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.datasets.data_utils import prepare_sample

import torch
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
import wandb


@registry.register_task("factvqa")
class FactVQATask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg): 
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )


    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        if is_main_process():
            wandb.log({
                'epoch': epoch
            })
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters #TODO: not affect loss_dict values for logging

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            if is_main_process():
                log_dict = {
                    f"train/{k}":v for k,v in loss_dict.items()
                }
                log_dict.update({"train/lr": optimizer.param_groups[0]["lr"]})
                wandb.log(log_dict)


        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        samples['prompt'] = samples['text_input']

        if 'image' not in samples.keys():
            samples['image'] = samples['img_path']

        img_ids = samples["image_id"]
        # logging.info(img_ids)
        
        answers = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        ## TODO load gt in val dataloader and calculate loss here

        for ans, img_id in zip(answers, img_ids):
            results.append({"answer": ans, "image_id": (img_id)})

        return results
    
    # @main_process
    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
            
        else:
            metrics = {"agg_metrics": -1}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        dictlist_res = json.load(open(eval_result_file, 'r'))
        logging.info(f'number: {len(dictlist_res)}')

        gt_label = []
        predict_label = []

        for one_dict in dictlist_res:
            plabel = self.extract_predict_label(one_dict['answer'])
            if plabel == 'yes':
                predict_label.append(1)
            elif plabel == 'no':
                predict_label.append(0)
            else:
                predict_label.append(2)
                logging.info(f"answer: {one_dict['answer']}, extracted_plabel: {plabel}")
            
            glabel = one_dict['image_id'].split('-')[1]
            gt_label.append(1 if glabel=='fake' else 0)


        metrics = self.classification_metrics(gt_label, predict_label)
        for key, value in metrics.items():
            logging.info(f"{key}: \n{value}\n")

        fields = ['accuracy', 'macro_f1']

        selected_metrics = {k: metrics[k] for k in metrics.keys() if k in fields}

        selected_metrics['agg_metrics'] = selected_metrics['accuracy']

        if split_name == 'val':
            log_dir = {
                f"val/{k}": v for k,v in selected_metrics.items()
            }
        elif split_name == 'test':
            log_dir = {
                f"test/{k}": v for k,v in selected_metrics.items()
            }
        else:
            print (f"error: {split_name}")
        
        if is_main_process():
            wandb.log(log_dir)

        return selected_metrics
    

    def extract_predict_label(self, sen):
        predict_label = sen.split(',')[0].lower().strip()
        
        if re.search(r'yes', predict_label, re.IGNORECASE):
            return 'yes'
        elif re.search(r'no', predict_label, re.IGNORECASE):
            return 'no'
        return None
    

    def classification_metrics(self, y_true, y_pred):

        unique_labels = set(y_true + y_pred)
        labels_to_consider = [label for label in unique_labels if label != 2]

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(y_true, y_pred, average=None, labels=labels_to_consider)
        recall = recall_score(y_true, y_pred, average=None, labels=labels_to_consider)
        f1 = f1_score(y_true, y_pred, average=None, labels=labels_to_consider)

        macro_precision = precision_score(y_true, y_pred, average='macro', labels=labels_to_consider)
        macro_recall = recall_score(y_true, y_pred, average='macro', labels=labels_to_consider)
        macro_f1 = f1_score(y_true, y_pred, average='macro', labels=labels_to_consider)

        micro_precision = precision_score(y_true, y_pred, average='micro', labels=labels_to_consider)
        micro_recall = recall_score(y_true, y_pred, average='micro', labels=labels_to_consider)
        micro_f1 = f1_score(y_true, y_pred, average='micro', labels=labels_to_consider)

        cm = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'confusion_matrix': cm
        }