import datetime
from functools import partial
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.training_and_loss import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch
import os

from tqdm import tqdm
from Segmentation import Segmentation
from utils.utils_metrics import compute_mIoU, show_results
from osgeo import gdal
from load_model import load_model

'''
训练自己的语义分割模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为png图片，无需固定大小，传入训练前会自动进行resize。
   由于许多同学的数据集是网络上下载的，标签格式并不符合，需要再度处理。一定要注意！标签的每个像素点的值就是这个像素点所属的种类。
   网上常见的数据集总共对输入图片分两类，背景的像素点值为0，目标的像素点值为255。这样的数据集可以正常运行但是预测是没有效果的！
   需要改成，背景的像素点值为0，目标的像素点值为1。
   如果格式有误，参考：https://github.com/bubbliiiing/segmentation-format-fix

2、训练好的权值文件保存在logs文件夹中，每个epoch都会保存一次，如果只是训练了几个step是不会保存的，epoch和step的概念要捋清楚一下。
   在训练过程中，该代码并没有设定只保存最低损失的，因此按默认参数训练完会有100个权值，如果空间不够可以自行删除。
   这个并不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一点，为了满足大多数的需求，还是都保存可选择性高。

3、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

4、调参是一门蛮重要的学问，没有什么参数是一定好的，现有的参数是我测试过可以正常训练的参数，因此我会建议用现有的参数。
   但是参数本身并不是绝对的，比如随着batch的增大学习率也可以增大，效果也会好一些；过深的网络不要用太大的学习率等等。
   这些都是经验上，只能靠各位同学多查询资料和自己试试了。
'''
if __name__ == "__main__":
    current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    print("当前时间:", current_time)
    # ------------------------------------------------------------------#
    #   model 选择模型
    #   可以选择SegFormer， Unet, TFBS, LSTM,
    #   放弃LSTMSegFormer，计算效率太低，而且精度并没有提升
    # ------------------------------------------------------------------#
    # models = ['TFBS', 'SegFormer', 'Unet', 'SegFormer_TLfromNE', 'Unet_TLfromNE', 'TFBS_TLfromNE']
    # models = ['SegFormer_TLfromNE', 'Unet_TLfromNE', 'TFBS_TLfromNE', 'SegFormer_TLfromRGB', 'Unet_TLfromRGB']
    models = ['TFBS']
    band = 'EVI'
    places = ['2020HT']
    training = True
    # transferlearning表示可以从别的地方的CDL预训练模型迁移
    transferlearning = False

    get_miou = True

    prediction = False

    lstm_outputses = [3]

    input_features = 1
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    for lstm_outputs in lstm_outputses:
        for place in places:
            for model_name in models:
                # -----------------------------------------------------#
                #   num_classes     训练自己的数据集必须要修改的
                #                   自己需要的分类个数+1，如2+1
                #                   NEofCHINA 3+1
                #                   2020HT 7+1
                #   get_miot = True 执行3_get_miou.py
                # -----------------------------------------------------#
                # pretrained表示加载基于RGB的预训练模型
                pretrained = True if 'TLfromRGB' in model_name else False
                # input_features = 12 if 'LSTM' in model_name or 'TFBS' in model_name or 'tae' in model_name else 72

                if transferlearning:
                    source_place = 'NEofCHINA'
                    source_numclass = 4
                    source_model = model_name.split('_')[0]
                    source_model_path = f'logs_{source_place}_{band}_{source_model}/logs/best_epoch_weights.pth'
                else:
                    source_numclass = None
                    source_model_path = None
                # -----------------------------------------------------#
                if place == '2020HT':
                    name_classes = ["Background", "Wheat", "Maize", "Tomato", "Sunflowers", "Squash", "Others"]
                if place == 'NEofCHINA':
                    name_classes = ["Others", "Maize", "Rice", "Soybean"]
                if place == 'IOWAofAMERICA':
                    name_classes = ['Others', 'Corn', 'Soybean']
                num_classes = len(name_classes)
                # ------------------------------------------------------------------#
                #   VOCdevkit_path  数据集路径
                # ------------------------------------------------------------------#
                VOCdevkit_path = os.path.join(r'J:\research\GEE\hetao_classification', place)

                if training:
                    #---------------------------------#
                    #   Cuda    是否使用Cuda
                    #           没有GPU可以设置成False
                    #---------------------------------#
                    Cuda            = True
                    #----------------------------------------------#
                    #   Seed    用于固定随机种子
                    #           使得每次独立训练都可以获得一样的结果
                    #----------------------------------------------#
                    seed            = 11
                    #---------------------------------------------------------------------#
                    #   distributed     用于指定是否使用单机多卡分布式运行
                    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
                    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
                    #   DP模式：
                    #       设置            distributed = False
                    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
                    #   DDP模式：
                    #       设置            distributed = True
                    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
                    #---------------------------------------------------------------------#
                    distributed     = False
                    #---------------------------------------------------------------------#
                    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
                    #---------------------------------------------------------------------#
                    sync_bn         = False
                    #---------------------------------------------------------------------#
                    #   fp16        是否使用混合精度训练
                    #               可减少约一半的显存、需要pytorch1.7.1以上
                    #---------------------------------------------------------------------#
                    fp16            = False
                    #-------------------------------------------------------------------#
                    #   所使用的的主干网络：
                    #   b0、b1、b2、b3、b4、b5
                    #-------------------------------------------------------------------#
                    phi             = "b5"
                    model_path      = ''
                    #------------------------------#
                    #   输入图片的大小
                    #------------------------------#
                    input_shape     = [128, 128]
                    # ------------------------------------------------------------------#
                    #   冻结阶段训练参数
                    #   此时模型的主干被冻结了，特征提取网络不发生改变
                    #   占用的显存较小，仅对网络进行微调
                    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
                    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
                    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
                    #                       （断点续练时使用）
                    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
                    #                       (当Freeze_Train=False时失效)
                    #   Freeze_batch_size   模型冻结训练的batch_size
                    #                       (当Freeze_Train=False时失效)
                    # ------------------------------------------------------------------#
                    Init_Epoch = 0
                    Freeze_Epoch = 50
                    Freeze_batch_size = 16
                    #------------------------------------------------------------------#
                    #   解冻阶段训练参数
                    #   此时模型的主干不被冻结了，特征提取网络会发生改变
                    #   占用的显存较大，网络所有的参数都会发生改变
                    #   UnFreeze_Epoch          模型总共训练的epoch
                    #   Unfreeze_batch_size     模型在解冻后的batch_size
                    #------------------------------------------------------------------#
                    UnFreeze_Epoch      = 30
                    Unfreeze_batch_size = 16
                    #------------------------------------------------------------------#
                    #   Freeze_Train    是否进行冻结训练
                    #                   默认先冻结主干训练后解冻训练。
                    #------------------------------------------------------------------#
                    Freeze_Train        = False
                    #------------------------------------------------------------------#
                    #   其它训练参数：学习率、优化器、学习率下降有关
                    #------------------------------------------------------------------#
                    #------------------------------------------------------------------#
                    #   Init_lr         模型的最大学习率
                    #                   当使用Adam优化器时建议设置      Init_lr=1e-4
                    #                   当使用AdamW优化器时建议设置     Init_lr=1e-4
                    #                   Transformer系列不建议使用SGD
                    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
                    #------------------------------------------------------------------#
                    Init_lr             = 1e-4
                    Min_lr              = Init_lr * 0.01
                    #------------------------------------------------------------------#
                    #   optimizer_type  使用到的优化器种类，可选的有adam、adamw、sgd
                    #   momentum        优化器内部使用到的momentum参数
                    #   weight_decay    权值衰减，可防止过拟合
                    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
                    #------------------------------------------------------------------#
                    optimizer_type      = "adamw"
                    momentum            = 0.9
                    weight_decay        = 1e-2
                    #------------------------------------------------------------------#
                    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
                    #------------------------------------------------------------------#
                    lr_decay_type       = 'cos'
                    #------------------------------------------------------------------#
                    #   save_period     多少个epoch保存一次权值
                    #------------------------------------------------------------------#
                    save_period         = 10
                    #------------------------------------------------------------------#
                    #   save_dir        权值与日志文件保存的文件夹
                    #------------------------------------------------------------------#
                    base = 'logs' + '_' + place + '_' + band + '_' + model_name + '_lstmOutputs' + str(lstm_outputs)
                    save_dir            = base +'/logs'
                    #------------------------------------------------------------------#
                    #   eval_flag       是否在训练时进行评估，评估对象为验证集
                    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
                    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
                    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
                    #   （一）此处获得的mAP为验证集的mAP。
                    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
                    #------------------------------------------------------------------#
                    eval_flag           = True
                    eval_period         = 2
                    #------------------------------------------------------------------#
                    #   建议选项：
                    #   种类少（几类）时，设置为True
                    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
                    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
                    #------------------------------------------------------------------#
                    dice_loss       = True
                    #------------------------------------------------------------------#
                    #   是否使用focal loss来防止正负样本不平衡
                    #------------------------------------------------------------------#
                    focal_loss      = False
                    #------------------------------------------------------------------#
                    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
                    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
                    #   如：
                    #   num_classes = 3
                    #   cls_weights = np.array([1, 2, 3], np.float32)
                    #------------------------------------------------------------------#
                    cls_weights     = np.ones([num_classes], np.float32)
                    #------------------------------------------------------------------#
                    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
                    #                   开启后会加快数据读取速度，但是会占用更多内存
                    #                   keras里开启多线程有些时候速度反而慢了许多
                    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
                    #------------------------------------------------------------------#
                    num_workers     = 1

                    seed_everything(seed)
                    #------------------------------------------------------#
                    #   设置用到的显卡
                    #------------------------------------------------------#
                    ngpus_per_node  = torch.cuda.device_count()
                    if distributed:
                        dist.init_process_group(backend="nccl")
                        local_rank  = int(os.environ["LOCAL_RANK"])
                        rank        = int(os.environ["RANK"])
                        device      = torch.device("cuda", local_rank)
                        if local_rank == 0:
                            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
                            print("Gpu Device Count : ", ngpus_per_node)
                    else:
                        device          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                        local_rank      = 0
                        rank            = 0

                    #----------------------------------------------------#
                    #   下载预训练权重
                    #----------------------------------------------------#
                    # SegFormer, Unet, TFBS, LSTM
                    model = load_model(model_name, input_features, num_classes, phi, lstm_outputs, pretrained,
                                       source_num=source_numclass, source_modelpath=source_model_path)

                    if not pretrained and not transferlearning:
                        weights_init(model)

                    #----------------------#
                    #   记录Loss
                    #----------------------#
                    if local_rank == 0:
                        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
                        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
                        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
                    else:
                        loss_history    = None

                    #------------------------------------------------------------------#
                    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
                    #   因此torch1.2这里显示"could not be resolve"
                    #------------------------------------------------------------------#
                    if fp16:
                        from torch.cuda.amp import GradScaler as GradScaler
                        scaler = GradScaler()
                    else:
                        scaler = None

                    model_train     = model.train()
                    #----------------------------#
                    #   多卡同步Bn
                    #----------------------------#
                    if sync_bn and ngpus_per_node > 1 and distributed:
                        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
                    elif sync_bn:
                        print("Sync_bn is not support in one gpu or not distributed.")

                    if Cuda:
                        if distributed:
                            #----------------------------#
                            #   多卡平行运行
                            #----------------------------#
                            model_train = model_train.cuda(local_rank)
                            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
                        else:
                            model_train = torch.nn.DataParallel(model)
                            cudnn.benchmark = True
                            model_train = model_train.cuda()

                    #---------------------------#
                    #   读取数据集对应的txt
                    #---------------------------#
                    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt"),"r") as f:
                        train_lines = f.readlines()
                    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt"),"r") as f:
                        val_lines = f.readlines()
                    num_train   = len(train_lines)
                    num_val     = len(val_lines)

                    if local_rank == 0:
                        show_config(
                            num_classes = num_classes, phi = phi, model_path = model_path, input_shape = input_shape, \
                            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
                            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
                            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
                        )
                        #---------------------------------------------------------#
                        #   总训练世代指的是遍历全部数据的总次数
                        #   总训练步长指的是梯度下降的总次数
                        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
                        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
                        #----------------------------------------------------------#
                        wanted_step = 1.5e4 if optimizer_type == "adamw" else 0.5e4
                        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
                        if total_step <= wanted_step:
                            if num_train // Unfreeze_batch_size == 0:
                                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
                            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
                            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
                            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
                            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

                    #------------------------------------------------------#
                    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
                    #   也可以在训练初期防止权值被破坏。
                    #   Init_Epoch为起始世代
                    #   Interval_Epoch为冻结训练的世代
                    #   Epoch总训练世代
                    #   提示OOM或者显存不足请调小Batch_size
                    #------------------------------------------------------#
                    if True:
                        UnFreeze_flag = False
                        #------------------------------------#
                        #   冻结一定部分训练
                        #------------------------------------#
                        if Freeze_Train:
                            for param in model.backbone.parameters():
                                param.requires_grad = False

                        #-------------------------------------------------------------------#
                        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
                        #-------------------------------------------------------------------#
                        # batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
                        batch_size = 8
                        #-------------------------------------------------------------------#
                        #   判断当前batch_size，自适应调整学习率
                        #-------------------------------------------------------------------#
                        nbs             = 16
                        lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
                        lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
                        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                        #---------------------------------------#
                        #   根据optimizer_type选择优化器
                        #---------------------------------------#
                        optimizer = {
                            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
                            'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
                            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
                        }[optimizer_type]

                        #---------------------------------------#
                        #   获得学习率下降的公式
                        #---------------------------------------#
                        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                        #---------------------------------------#
                        #   判断每一个世代的长度
                        #---------------------------------------#
                        epoch_step      = num_train // batch_size
                        epoch_step_val  = num_val // batch_size

                        if epoch_step == 0 or epoch_step_val == 0:
                            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                        train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path, band)
                        val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path, band)

                        if distributed:
                            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
                            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
                            batch_size      = batch_size // ngpus_per_node
                            shuffle         = False
                        else:
                            train_sampler   = None
                            val_sampler     = None
                            shuffle         = True

                        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                                    drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler,
                                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                                    drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler,
                                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                        #----------------------#
                        #   记录eval的map曲线
                        #----------------------#
                        if local_rank == 0:
                            eval_callback   = EvalCallback(model, input_shape, band, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                                            eval_flag=eval_flag, period=eval_period)
                        else:
                            eval_callback   = None

                        #---------------------------------------#
                        #   开始模型训练
                        #---------------------------------------#
                        for epoch in range(Init_Epoch, UnFreeze_Epoch):
                            #---------------------------------------#
                            #   如果模型有冻结学习部分
                            #   则解冻，并设置参数
                            #---------------------------------------#
                            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                                batch_size = Unfreeze_batch_size

                                #-------------------------------------------------------------------#
                                #   判断当前batch_size，自适应调整学习率
                                #-------------------------------------------------------------------#
                                nbs             = 16
                                lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
                                lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
                                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                                #---------------------------------------#
                                #   获得学习率下降的公式
                                #---------------------------------------#
                                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                                for param in model.backbone.parameters():
                                    param.requires_grad = True

                                epoch_step      = num_train // batch_size
                                epoch_step_val  = num_val // batch_size

                                if epoch_step == 0 or epoch_step_val == 0:
                                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                                            drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler,
                                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                                            drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler,
                                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                                UnFreeze_flag   = True

                            if distributed:
                                train_sampler.set_epoch(epoch)

                            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

                            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, \
                                dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

                            if distributed:
                                dist.barrier()

                        if local_rank == 0:
                            loss_history.writer.close()

                # 执行3_get_miou.py
                if get_miou:
                    # -----------------------------------------------------#
                    #   num_classes     训练自己的数据集必须要修改的
                    #                   自己需要的分类个数+1，如2+1
                    #                   NEofCHINA 3+1
                    #                   hetao 7+1
                    #                   IOWAofAMERICA 2+1
                    # 含有LSTM模块的input_features为12
                    # -----------------------------------------------------#
                    model_path = base +'/logs/last_epoch_weights.pth'

                    image_ids = open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
                    gt_dir = os.path.join(VOCdevkit_path, "SegmentationClass/")
                    miou_out_path = base +"/miou_out"
                    pred_dir = os.path.join(miou_out_path, 'detection-results')

                    if miou_mode == 0 or miou_mode == 1:
                        pretrained = False
                        if not os.path.exists(pred_dir):
                            os.makedirs(pred_dir)

                        print("get_miou Load model.")
                        model = SegFormer_Segmentation(input_features=input_features, num_classes=num_classes,
                                                       lstm_outputs=lstm_outputs, model=model_name.split('_')[0],
                                                       pretrained=pretrained, model_path=model_path)
                        print("Load model done.")

                        print("Get predict result.")
                        for image_id in tqdm(image_ids):
                            image_path = os.path.join(os.path.join(VOCdevkit_path, band), image_id + ".tif")
                            image = gdal.Open(image_path)
                            image = model.get_miou_png(image)
                            image.save(os.path.join(pred_dir, image_id + ".tif"))
                        print("Get predict result done.")

                    if miou_mode == 0 or miou_mode == 2:
                        print("Get miou.")
                        hist, IoUs, PA_Recall, Precision, Overall_acc = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                                        name_classes)  # 执行计算mIoU的函数
                        print("Get miou done.")
                        show_results(hist, miou_out_path, IoUs, PA_Recall, Precision, Overall_acc, name_classes)

                if prediction:
                    from utils.utils import merge_tifs
                    from PIL import Image
                    import shutil

                    pretrained = False
                    print('prediction start: Load model.')
                    images_for_prediction_path = os.path.join(VOCdevkit_path, band+"_forPredict/")
                    prediction_save_path = base + '/prediction_images_pieces'
                    middle_tif_path = base + '/prediction_middle.tif'
                    new_tif_path = base +  '/prediction.tif'
                    model_path = base +  '/logs/last_epoch_weights.pth'
                    if place == '2020HT':
                        reference_file =r'J:\research\GEE\hetao_classification\2020HT\htCDL-rf\CDL\RF2020HTclassification_3from2.tif'
                    if place == 'NEofCHINA':
                        reference_file = r'J:\research\GEE\hetao_classification\NEofCHINA\NEofCHINA_2019CDL\CDL\NEofCHINA_crop_2019cdl.tif'
                    if place == 'IOWAofAMERICA':
                        reference_file = r'J:\research\GEE\hetao_classification\IOWAofAMERICA\usa_cdl\CDL\AmerciaCDL_2020-10m.tif'
                    model = SegFormer_Segmentation(input_features=input_features, num_classes=num_classes,
                                                   lstm_outputs=lstm_outputs, model=model_name.split('_')[0],
                                                   pretrained=pretrained, model_path=model_path)

                    img_names = os.listdir(images_for_prediction_path)
                    for img_name in tqdm(img_names):
                        if img_name.lower().endswith(('.tif')):
                            image_path = os.path.join(images_for_prediction_path, img_name)
                            image = gdal.Open(image_path)

                            image_array = image.ReadAsArray()
                            # Check if the image is all zeros
                            if np.all(image_array == 0):
                                # If all zeros, return a zero matrix with the same dimensions
                                x, y = image_array.shape[1], image_array.shape[2]
                                r_image = np.zeros((x, y, 3))
                                r_image = Image.fromarray(np.uint8(r_image))
                            else:
                                # If not all zeros, run the model to detect features
                                r_image = model.detect_image(image)

                            if not os.path.exists(prediction_save_path):
                                os.makedirs(prediction_save_path)
                            r_image.save(os.path.join(prediction_save_path, img_name))

                    merge_tifs(prediction_save_path, reference_file, middle_tif_path, new_tif_path)
                    os.remove(middle_tif_path)
                    shutil.rmtree(prediction_save_path)

    current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    print("当前时间:", current_time)
