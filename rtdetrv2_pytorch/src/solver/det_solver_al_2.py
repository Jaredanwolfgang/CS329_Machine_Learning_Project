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


"""
There are 7480 images in kitti dataset
The training procedure have 72 epochs
Initially 1500 data in the training set
add 1500 data in epoch #10, 20, 30, 40 
"""

AL_MODE = 'random'
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

class DetSolver(BaseSolver):
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'Number of trainable parameters: {n_parameters}')
        best_stat = {'epoch': -1, }
        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        if dist.get_rank() == 0:
            kitti_dict = clean_file_structure()
            all_train_img_ids = [_['id'] for _ in kitti_dict['images']]
            all_unlabeled_img_ids = [_ for _ in all_train_img_ids]
            all_labeled_img_ids = []
        dist.barrier()

        for epoch in range(start_epcoch, args.epoches):
            # Evaluate the model and sort the image based on their entropy
            for i in range(dist.get_world_size()):
                locals()[f'pairs_gpu{i}'] = []
            dist.barrier()
            if epoch % AL_EPOCH_MARGIN == 0 and AL_MODE == 'entropy':
                print(f"[AL] Pausing at epoch {epoch}. Precomputation for entropy strtegy ...")
                #-----------------------Unlabeled data loader-----------------------#
                self.cfg.labeled_dataloader = self.cfg.build_dataloader('unlabeled_dataloader')
                self.unlabeled_dataloader = dist_utils.warp_loader(self.cfg.unlabeled_dataloader, \
                    shuffle=self.cfg.unlabeled_dataloader.shuffle)
                self.unlabeled_dataloader.set_epoch(epoch)
                if dist_utils.is_dist_available_and_initialized():
                    self.unlabeled_dataloader.sampler.set_epoch(epoch)
                #-------------------------------------------------------------------#
                self.model.eval()
                self.criterion.eval()
                dist.barrier()
                # with torch.no_grad():
                #     locals()[f'pairs_gpu{dist.get_rank()}'] = self.entropy_feed()
                #     print(len(locals()[f'pairs_gpu{dist.get_rank()}']))
                # dist.barrier()
                # pairs = []
                # all_pairs_gpu = torch.tensor(locals()[f'pairs_gpu{dist.get_rank()}']).to('cuda:0')
                # dist.all_gather([pairs, all_pairs_gpu])
                # dist.barrier()
                # print(-len(pairs))
                # if dist.get_rank() == 0:
                #     for i in range(dist.get_world_size()):
                #         pairs += locals()[f'pairs_gpu{i}']
                #         print(len(locals()[f'pairs_gpu{i}']))
                #     pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
                #     print(f"[AL] Precomputation done, collected {len(pairs)} pairs using entropy.")
                #     assert False
                with torch.no_grad():
                    local_pairs = self.entropy_feed()
                    rank = dist.get_rank()
                    local_pairs_tensor = torch.tensor(local_pairs).to(torch.device("cuda", rank))
                    all_pairs = [torch.zeros_like(local_pairs_tensor) for _ in range(dist.get_world_size())]
                    dist.all_gather(all_pairs, local_pairs_tensor)
                    if rank == 0:
                        pairs = []
                        for i in range(dist.get_world_size()):
                            pairs.extend(all_pairs[i].cpu().numpy())
                        pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
                        print(f"[AL] Precomputation done, collected {len(pairs)} pairs using entropy.")
                    dist.barrier()
                
            if epoch % AL_EPOCH_MARGIN == 0 and dist.get_rank() == 0:
                print(f"[AL] Pausing at epoch {epoch}. Selecting data to label ...")
                # all_train_data = os.listdir(os.path.join(KITTI_DATASET_PATH, 'train2017'))
                # all_train_data = [_.split('.')[0] for _ in all_train_data]
                # all_labeled_data = os.listdir(os.path.join(LABELED_DATASET_PATH, 'train2017'))
                # all_labeled_data = [_.split('.')[0] for _ in all_labeled_data]
                # all_unlabeled_data = [_ for _ in all_train_data if _ not in all_labeled_data]
                
                if AL_MODE == 'random':
                    new_labeled_img_ids = AL_random(all_unlabeled_img_ids)
                elif AL_MODE == 'entropy':
                    new_labeled_img_ids = [_[1] for _ in pairs]
                    new_labeled_img_ids = getTopUnique(new_labeled_img_ids, AL_SIZE_MARGIN, all_labeled_img_ids)
                else:
                    raise NotImplementedError
                    
                print(f"[AL] Selected {len(new_labeled_img_ids)} data from unlabeled dataset.")
                print("[AL] Labeling data ...")
                # for datum in tqdm(new_labeled_img_ids):
                #     img_file_src = os.path.join(UNLABELED_DATASET_PATH, 'train2017', f"{datum}.png")
                #     img_file_dst = os.path.join(LABELED_DATASET_PATH, 'train2017')
                #     shutil.move(img_file_src, img_file_dst)
                if len(new_labeled_img_ids) > 0:
                    all_labeled_img_ids += new_labeled_img_ids
                    all_unlabeled_img_ids = [_ for _ in all_unlabeled_img_ids if _ not in new_labeled_img_ids]
                    with open(os.path.join(KITTI_DATASET_PATH, 'annotations', 'instances_labeled2017.json'), 'w') as f:
                        labeled_ann_data = {
                            'images': [_ for _ in kitti_dict['images'] if _['id'] in all_labeled_img_ids],
                            'annotations': [_ for _ in kitti_dict['annotations'] if _['image_id'] in all_labeled_img_ids],
                            'categories': kitti_dict['categories']
                        }
                        print(f"[AL DEBUG] {len(all_labeled_img_ids)}, {len(all_unlabeled_img_ids)}, {len(labeled_ann_data['images'])}")
                        json.dump(labeled_ann_data, f)
                    with open(os.path.join(KITTI_DATASET_PATH, 'annotations', 'instances_unlabeled2017.json'), 'w') as f:
                        unlabeled_ann_data = {
                            'images': [_ for _ in kitti_dict['images'] if _['id'] in all_unlabeled_img_ids],
                            'annotations': [_ for _ in kitti_dict['annotations'] if _['image_id'] in all_unlabeled_img_ids],
                            'categories': kitti_dict['categories']
                        }
                        json.dump(unlabeled_ann_data, f)
                    # all_ann_file = os.path.join(KITTI_DATASET_PATH, 'annotations', 'instances_train2017.json')
                    # with open(all_ann_file, 'r') as f_ann_in:
                    #     all_ann_data = json.load(f_ann_in)
                    # labeled_ann_file = os.path.join(LABELED_DATASET_PATH, 'annotations', 'instances_train2017.json')
                    # with open(labeled_ann_file, 'r') as f_ann_in:
                    #     labeled_ann_data = json.load(f_ann_in)
                    # new_filenames = [_ + '.png' for _ in new_labeled_data]
                    # all_ann_data['images'] = [_ for _ in all_ann_data['images'] if _['file_name'] in new_filenames]  # 1500
                    # labeled_ann_data['images'] += all_ann_data['images']  # 1500 * (n + 1)
                    # labeled_image_ids = [_['id'] for _ in labeled_ann_data['images']]  # 1500 * (n + 1)
                    # labeled_ann_data['annotations'] = [_ for _ in all_ann_data['annotations'] if _['image_id'] in labeled_image_ids]
                    # ann_file = os.path.join(LABELED_DATASET_PATH, 'annotations', 'instances_train2017.json')
                    # with open(ann_file, 'w') as f_ann_out:
                    #     json.dump(labeled_ann_data, f_ann_out)

                    # ann_file = os.path.join(UNLABELED_DATASET_PATH, 'annotations', 'instances_train2017.json')
                    # with open(ann_file, 'r') as f_ann_in:
                    #     ann_data = json.load(f_ann_in)
                    # new_filenames = [_ + '.png' for _ in new_labeled_data]  # 1500
                    # ann_data['images'] = [_ for _ in ann_data['images'] if _['file_name'] not in new_filenames]  # 1500 * (n - 1)
                    # unlabeled_image_ids = [_['id'] for _ in ann_data['images']]  # 1500 * (n - 1)
                    # ann_data['annotations'] = [_ for _ in ann_data['annotations'] if _['image_id'] in unlabeled_image_ids]
                    # ann_file = os.path.join(UNLABELED_DATASET_PATH, 'annotations', 'instances_train2017.json')
                    # with open(ann_file, 'w') as f_ann_out:
                    #     json.dump(ann_data, f_ann_out)
            dist.barrier()
            
            #-----------------------Labeled data loader-----------------------#
            if epoch % AL_EPOCH_MARGIN == 0:
                print(f"[AL][{dist.get_rank()}] Reconstructing dataloader ...")
                self.cfg.labeled_dataloader = self.cfg.build_dataloader('labeled_dataloader')
                self.labeled_dataloader = dist_utils.warp_loader(self.cfg.labeled_dataloader, \
                    shuffle=self.cfg.labeled_dataloader.shuffle)
                # print(len(self.train_dataloader), len(self.labeled_dataloader))
                print(f"[AL][{dist.get_rank()}] done with {len(self.labeled_dataloader)} in the new dataloader!")
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
    
    def entropy_feed(self):
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
        # sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
        # print(f"[AL] {len(sorted_pairs)} pairs collected here in {self.device}.")
        return pairs


def clean_file_structure():
    print("[AL] Updating Files")
    # if os.path.exists(LABELED_DATASET_PATH):
    #     shutil.rmtree(LABELED_DATASET_PATH)
    # os.makedirs(LABELED_DATASET_PATH)
    # os.makedirs(os.path.join(LABELED_DATASET_PATH, 'train2017'))
    # os.makedirs(os.path.join(LABELED_DATASET_PATH, 'annotations'))
    base_ann = {
        'images': [],
        'annotations': [],
        "categories": [
            {"id": 1, "name": "Car"},
            {"id": 2, "name": "Pedestrian"},
            {"id": 3, "name": "Cyclist"}
        ]
    }
    with open(os.path.join(KITTI_DATASET_PATH, 'annotations', 'instances_labeled2017.json'), 'w') as f:
        json.dump(base_ann, f)

    # if os.path.exists(UNLABELED_DATASET_PATH):
    #     shutil.rmtree(UNLABELED_DATASET_PATH)
    # os.makedirs(UNLABELED_DATASET_PATH)
    # os.makedirs(os.path.join(UNLABELED_DATASET_PATH, 'train2017'))
    # os.makedirs(os.path.join(UNLABELED_DATASET_PATH, 'annotations'))
    shutil.copy2(
        os.path.join(KITTI_DATASET_PATH, 'annotations', 'instances_train2017.json'),
        os.path.join(KITTI_DATASET_PATH, 'annotations', 'instances_unlabeled2017.json')
    )
    # src_folder = os.path.join(KITTI_DATASET_PATH, 'train2017')
    # dst_folder = os.path.join(UNLABELED_DATASET_PATH, 'train2017')
    # for file in tqdm(os.listdir(src_folder)):
    #     shutil.copy2(
    #         os.path.join(src_folder, file),
    #         os.path.join(dst_folder, file)
    #     )
    with open(os.path.join(KITTI_DATASET_PATH, 'annotations', 'instances_train2017.json'), 'r') as f:
        kitti_dict = json.load(f)
    return kitti_dict

def getTopUnique(lst: list, topK: int, xclude: list = []):
    res  = []
    for ele in lst:
        if ele not in res and ele not in xclude:
            res.append(ele)
        if len(res) >= topK:
            break
    return res
