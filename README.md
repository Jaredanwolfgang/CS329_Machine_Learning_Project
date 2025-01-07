# CS329_Machine_Learning_Project
When Active Learning and Data Augmentation meet at Object Detection

The dataset we use is downloaded from [here](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). We then use [kitti2coco](https://github.com/kouyuanbo/kitti2coco) to convert the dataset to COCO format.

The original model is [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch) and we only save the `rtdetrv2_pytorch` folder in this repository.

The original test video is recorded by ourselves in the campus:


https://github.com/user-attachments/assets/9e9f5bcd-8909-46df-b81a-8873589dd4c8

The detection result using the original model is:

https://github.com/user-attachments/assets/12e467d8-ece9-4ff6-a1a1-f8f47650d66a

And our finetune result is:

https://github.com/user-attachments/assets/08d07829-59dc-456c-9e9d-379cad483967

You can see the promising tuning results help us distinguish the pedestrian and cyclist. (Using the original model will somehow misunderstand the cyclist as person) During the finetune process, we have used the checkpoint `rtdetrv2_r101vd_6x_coco_from_paddle.pth` from the original model. And then we use Active Learning and Data Augmentation to finetune the model. 
