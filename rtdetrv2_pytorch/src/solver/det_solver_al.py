"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate

import os
import shutil
import random
from tqdm import tqdm

from ..data import KittiData
from .nets import Net


"""
There are 7480 images in kitti dataset
The training procedure have 72 epochs
Initially 1500 data in the training set
add 1500 data in epoch #10, 20, 30, 40 
"""

AL_MODE = 'entropy'
AL_SIZE_MARGIN = 1500
AL_EPOCH_MARGIN = 5
AL_DATASET_BASE = os.path.join('.', 'dataset')
LABELED_DATASET_PATH = os.path.join(AL_DATASET_BASE, 'kitti_labeled')
UNLABELED_DATASET_PATH = os.path.join(AL_DATASET_BASE, 'kitti_unlabeled')
KITTI_DATASET_PATH = os.path.join(AL_DATASET_BASE, 'kitti_coco')

def AL_random(unlabeled_data):
    if len(unlabeled_data) < AL_SIZE_MARGIN:
        return unlabeled_data
    return random.sample(unlabeled_data, AL_SIZE_MARGIN)

class ALDetSolver(BaseSolver):
    def load_data(self, train_transform=None, val_transform=None):
        self.data =  KittiData(
            root=KITTI_DATASET_PATH,
            annTrainFile=os.path.join(KITTI_DATASET_PATH, 'annotations', 'instances_train2017.json'),
            annTestFile=os.path.join(KITTI_DATASET_PATH, 'annotations', 'instances_val2017.json'),
            train_transform=train_transform,
            val_transform=val_transform
        )
    
    def init_net(self, ):
        self.net = Net(self.model, self.device)
        
    def count_params(self, ):
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'Number of trainable parameters: {n_parameters}')
    
    def fit(self, ):
        print("Training parameters initializing...")
        # By calling the train() function, the parameters are initialized.
        # But the model is not trained yet.
        self.train() 
        args = self.cfg
        
        # Load the data
        self.load_data(args.train_dataloader.dataset.transforms, args.val_dataloader.dataset.transforms)

        # Initialize the network
        self.init_net()

        # Count the number of trainable parameters
        self.count_params()
        
        best_stat = {'epoch': -1, }
        start_time = time.time()
        start_epcoch = self.last_epoch + 1

        for epoch in range(start_epcoch, args.epoches):
            # Evaluate the model and sort the image based on their entropy
            pairs = []
            if epoch % AL_EPOCH_MARGIN == 0 and AL_MODE == 'entropy':
                print(f"[AL] Pausing at epoch {epoch}. Precomputation for entropy strtegy ...")
                #-----------------------Unlabeled data loader-----------------------#
                self.unlabeled_dataloader = dist_utils.warp_loader(self.cfg.unlabeled_dataloader, \
                    shuffle=self.cfg.unlabeled_dataloader.shuffle)
                self.unlabeled_dataloader.set_epoch(epoch)
                if dist_utils.is_dist_available_and_initialized():
                    self.unlabeled_dataloader.sampler.set_epoch(epoch)
                #-------------------------------------------------------------------#
                self.model.eval()
                self.criterion.eval()
                dist.barrier()
                with torch.no_grad():
                    pairs = self.gather_pairs_and_compute_entropy()
                dist.barrier()
                print(f"[AL] Precomputation done!")
                
            if epoch % AL_EPOCH_MARGIN == 0 and dist.get_rank() == 0:
                print(f"[AL] Pausing at epoch {epoch}. Selecting data to label ...")
                all_train_data = os.listdir(os.path.join(KITTI_DATASET_PATH, 'train2017'))
                all_train_data = [_.split('.')[0] for _ in all_train_data]
                all_labeled_data = os.listdir(os.path.join(LABELED_DATASET_PATH, 'train2017'))
                all_labeled_data = [_.split('.')[0] for _ in all_labeled_data]
                all_unlabeled_data = [_ for _ in all_train_data if _ not in all_labeled_data]
                
                if AL_MODE == 'random':
                    new_labeled_data = AL_random(all_unlabeled_data)
                elif AL_MODE == 'entropy':
                    print(f"[AL] Collected {len(pairs)} pairs using entropy.")
                    new_image_ids = pairs[:AL_SIZE_MARGIN]
                    new_image_ids = [_[1] for _ in new_image_ids]
                    new_labeled_data = [_['file_name'].split('.')[0] for _ in kitti_dict['images'] if _['id'] in new_image_ids]
                else:
                    new_labeled_data = AL_random(all_unlabeled_data)
                    
                print(f"[AL] Selected {len(new_labeled_data)} data from unlabeled dataset.")
                print("[AL] Labeling data ...")
                for datum in tqdm(new_labeled_data):
                    img_file_src = os.path.join(UNLABELED_DATASET_PATH, 'train2017', f"{datum}.png")
                    img_file_dst = os.path.join(LABELED_DATASET_PATH, 'train2017')
                    shutil.move(img_file_src, img_file_dst)
                if len(new_labeled_data) > 0:
                    all_ann_file = os.path.join(KITTI_DATASET_PATH, 'annotations', 'instances_train2017.json')
                    with open(all_ann_file, 'r') as f_ann_in:
                        all_ann_data = json.load(f_ann_in)
                    labeled_ann_file = os.path.join(LABELED_DATASET_PATH, 'annotations', 'instances_train2017.json')
                    with open(labeled_ann_file, 'r') as f_ann_in:
                        labeled_ann_data = json.load(f_ann_in)
                    new_filenames = [_ + '.png' for _ in new_labeled_data]
                    all_ann_data['images'] = [_ for _ in all_ann_data['images'] if _['file_name'] in new_filenames]  # 1500
                    labeled_ann_data['images'] += all_ann_data['images']  # 1500 * (n + 1)
                    labeled_image_ids = [_['id'] for _ in labeled_ann_data['images']]  # 1500 * (n + 1)
                    labeled_ann_data['annotations'] = [_ for _ in all_ann_data['annotations'] if _['image_id'] in labeled_image_ids]
                    ann_file = os.path.join(LABELED_DATASET_PATH, 'annotations', 'instances_train2017.json')
                    with open(ann_file, 'w') as f_ann_out:
                        json.dump(labeled_ann_data, f_ann_out)

                    ann_file = os.path.join(UNLABELED_DATASET_PATH, 'annotations', 'instances_train2017.json')
                    with open(ann_file, 'r') as f_ann_in:
                        ann_data = json.load(f_ann_in)
                    new_filenames = [_ + '.png' for _ in new_labeled_data]  # 1500
                    ann_data['images'] = [_ for _ in ann_data['images'] if _['file_name'] not in new_filenames]  # 1500 * (n - 1)
                    unlabeled_image_ids = [_['id'] for _ in ann_data['images']]  # 1500 * (n - 1)
                    ann_data['annotations'] = [_ for _ in ann_data['annotations'] if _['image_id'] in unlabeled_image_ids]
                    ann_file = os.path.join(UNLABELED_DATASET_PATH, 'annotations', 'instances_train2017.json')
                    with open(ann_file, 'w') as f_ann_out:
                        json.dump(ann_data, f_ann_out)
            dist.barrier()
            
            #-----------------------Labeled data loader-----------------------#
            if epoch % AL_EPOCH_MARGIN == 0:
                print(f"[AL][{dist.get_rank()}] Reconstructing dataloader ...")
                self.labeled_dataloader = dist_utils.warp_loader(self.cfg.labeled_dataloader, \
                    shuffle=self.cfg.labeled_dataloader.shuffle)
                # print(len(self.train_dataloader), len(self.labeled_dataloader))
                print(f"[AL][{dist.get_rank()}] Done!")
            self.labeled_dataloader.set_epoch(epoch)
            # self.labeled_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.labeled_dataloader.sampler.set_epoch(epoch)
            #-------------------------------------------------------------------#
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.labeled_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader, 
                self.evaluator, 
                self.device
            )

            # TODO 
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
            
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
    
    def gather_pairs_and_compute_entropy(self):
        """
            return: list([entropy, image_id])
        """
        pairs = []
        for samples, targets in tqdm(self.unlabeled_dataloader):
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            outputs = self.model(samples)
            for (output, target) in zip(outputs['pred_logits'], targets):
                output = output.view(-1)
                probs = F.softmax(output, dim=-1)
                log_probs = F.log_softmax(output, dim=-1)
                # The pairs store the information of the entropy and the according image. (Target here is a dict)
                pairs.append([-torch.sum(probs * log_probs).item(), target['image_id']])
        pairs_tensor = torch.tensor(pairs, dtype=torch.float32, device=self.device)
        world_size = dist.get_world_size()
        all_pairs = [torch.empty_like(pairs_tensor) for _ in range(world_size)]
        dist.all_gather(all_pairs, pairs_tensor)
        gathered_pairs = torch.cat(all_pairs, dim=0).cpu().tolist()
        sorted_pairs = sorted(gathered_pairs, key=lambda x: x[0], reverse=True)
        return sorted_pairs


def clean_file_structure():
    print("[AL] Updating Files")
    if os.path.exists(LABELED_DATASET_PATH):
        shutil.rmtree(LABELED_DATASET_PATH)
    os.makedirs(LABELED_DATASET_PATH)
    os.makedirs(os.path.join(LABELED_DATASET_PATH, 'train2017'))
    os.makedirs(os.path.join(LABELED_DATASET_PATH, 'annotations'))
    base_ann = {
        'images': [],
        'annotations': [],
        "categories": [
            {"id": 1, "name": "Car"},
            {"id": 2, "name": "Pedestrian"},
            {"id": 3, "name": "Cyclist"}
        ]
    }
    with open(os.path.join(LABELED_DATASET_PATH, 'annotations', 'instances_train2017.json'), 'w') as f:
        json.dump(base_ann, f)

    if os.path.exists(UNLABELED_DATASET_PATH):
        shutil.rmtree(UNLABELED_DATASET_PATH)
    os.makedirs(UNLABELED_DATASET_PATH)
    os.makedirs(os.path.join(UNLABELED_DATASET_PATH, 'train2017'))
    os.makedirs(os.path.join(UNLABELED_DATASET_PATH, 'annotations'))
    shutil.copy2(
        os.path.join(KITTI_DATASET_PATH, 'annotations', 'instances_train2017.json'),
        os.path.join(UNLABELED_DATASET_PATH, 'annotations', 'instances_train2017.json')
    )
    src_folder = os.path.join(KITTI_DATASET_PATH, 'train2017')
    dst_folder = os.path.join(UNLABELED_DATASET_PATH, 'train2017')
    for file in tqdm(os.listdir(src_folder)):
        shutil.copy2(
            os.path.join(src_folder, file),
            os.path.join(dst_folder, file)
        )
    with open(os.path.join(KITTI_DATASET_PATH, 'annotations', 'instances_train2017.json'), 'r') as f:
        kitti_dict = json.load(f)
    return kitti_dict