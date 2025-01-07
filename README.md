# CS329_Machine_Learning_Project
When Active Learning and Data Augmentation meet at Object Detection

The dataset we use is downloaded from [here](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). We then use [kitti2coco](https://github.com/kouyuanbo/kitti2coco) to convert the dataset to COCO format.

The original model is [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch) and we only save the `rtdetrv2_pytorch` folder in this repository.

The original test video is recorded by ourselves in the campus:

![test](https://github.com/Jaredanwolfgang/CS329_Machine_Learning_Project/blob/main/video/test_clipped.mp4)

The detection result using the original model is:

![original](https://github.com/Jaredanwolfgang/CS329_Machine_Learning_Project/blob/main/video/output_original_clipped.mp4)

And our finetune result is:

![final](https://github.com/Jaredanwolfgang/CS329_Machine_Learning_Project/blob/main/video/output_al_clipped.mp4)

During the finetune process, we have used the checkpoint `rtdetrv2_r101vd_6x_coco_from_paddle.pth` from the original model. And then we use Active Learning and Data Augmentation to finetune the model. 
