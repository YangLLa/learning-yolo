import datetime
import argparse

import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from models import *
from build_utils.datasets import *
from build_utils.utils import *
from train_utils import train_eval_utils as train_util
from train_utils import get_coco_api_from_dataset


def train(hyp):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    wdir = "weights" + os.sep  # weights dir
    best = wdir + "best.pt"
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    # 由于显存的原因batchsize不能设置的很大，可以在更新多轮之后再更新一次权重，有助于模型训练
    # 每迭代64/batch_size次更新一下参数
    weights = opt.weights  # initial training weights
    imgsz_train = opt.img_size
    imgsz_test = opt.img_size  # test image sizes
    multi_scale = opt.multi_scale

    # Image sizes
    # 图像要设置成32的倍数
    gs = 64  # 32 (pixels) grid size   608/32 = 19
    assert math.fmod(imgsz_test, gs) == 0, "--img-size %g must be a %g-multiple" % (imgsz_test, gs)
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs
    if multi_scale:
        imgsz_min = opt.img_size // 1.5
        imgsz_max = opt.img_size // 0.667

        # 将给定的最大，最小输入尺寸向下调整到32的整数倍
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max  # initialize with max size
        print("Using multi_scale training, image range[{}, {}]".format(imgsz_min, imgsz_max))

    # configure run
    init_seeds(0)  # 初始化随机种子，保证结果可复现
    data_dict = parse_data_cfg(data)
    train_path = data_dict["train"]
    test_path = data_dict["valid"]
    nc = 1 if opt.single_cls else int(data_dict["classes"])  # number of classes
    hyp["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset
    hyp["obj"] *= imgsz_test / 320

    # Remove previous results
    for f in glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg).to(device)


    # 是否冻结权重，只训练predictor的权重
    if opt.freeze_layers:
        # 索引减一对应的是predictor的索引，YOLOLayer并不是predictor
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if
                                isinstance(module, YOLOLayer)]
        # 冻结除predictor和YOLOLayer外的所有层
        freeze_layer_indeces = [x for x in range(len(model.module_list)) if
                                (x not in output_layer_indices) and
                                (x - 1 not in output_layer_indices)]
        # Freeze non-output layers
        # 总共训练3x2=6个parameters
        for idx in freeze_layer_indeces:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    # else:
    #     # 如果freeze_layer为False，默认仅训练除darknet53之后的部分
    #     # 若要训练全部权重，删除以下代码
    #     darknet_end_layer = 74  # only yolov3spp cfg   74
    #     # Freeze darknet53 layers
    #     # 总共训练21x3+3x2=69个parameters
    #     for idx in range(darknet_end_layer + 1):  # [0, 74]
    #         for parameter in model.module_list[idx].parameters():
    #             parameter.requires_grad_(False)

        #print("冻结了基础权重")pyhon

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=hyp["lr0"], momentum=hyp["momentum"],
                          weight_decay=hyp["weight_decay"], nesterov=True)
    # optimizer = optim.Adam(pg, lr=hyp["lr0"],betas=(0.9,0.999),eps=1e-8,
    #                        weight_decay=hyp["weight_decay"],amsgrad=False)
    #optimizer = torch.optim.Adam(pg, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
    # weight_decay=0, amsgrad=False, *, maximize=False)

    #optimizer = optim.RMSprop(pg, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    scaler = torch.cuda.amp.GradScaler() if opt.amp else None
    print("scaler: ",scaler)

    start_epoch = 0
    best_map = 0.0
    if weights.endswith(".pt") or weights.endswith(".pth"):
        # 先进行读取权重
        ckpt = torch.load(weights, map_location=device)
        # load model
        try:
            # 载入预训练模型ckpt及其权重(ckpt['model'])  a.numel() a张量中元素的总数  a.shape() a张量的形状
            #ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
            # #当载入的cfg网络框架与权重完全一致时使用

            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if k in model.state_dict() and (v.shape == model.state_dict()[k].shape)}
            # #判断预训练模型中网络的模块是否在修改后的网络中也存在，当存在且shape相同时就取出放在ckpt中
            #
            model.state_dict().update(ckpt["model"])    # 权重更新，更新为修改之后的
            model.load_state_dict(ckpt["model"], strict=False)  # 加载想要的权重进入网络
            # missing_keys, unexpected_keys = model.load_state_dict(ckpt["model"], strict=False)
            #
            # print("在构建的网络模型中有一部分并没有在预训练权重中出现[missing_keys]:",*missing_keys,sep="\n")
            # print("[unexpected_keys]:", *unexpected_keys, sep="\n")

            # model_dict = model.state_dict()  # 得到我们模型的参数
            #
            # # 判断预训练模型中网络的模块是否修改后的网络中也存在，并且shape相同，如果相同则取出
            # pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict and (v.shape == model_dict[k].shape)}
            #
            # # 更新修改之后的 model_dict
            # model_dict.update(pretrained_dict)

            # 加载我们真正需要的 state_dict
            # model.load_state_dict(model_dict, strict=False)
            print("预训练权重读取")
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" \
                % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        #load optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            if "best_map" in ckpt.keys():
                best_map = ckpt["best_map"]


        # load results
        if ckpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_results"])  # write results.txt

        # epochs
        # start_epoch = ckpt["epoch"] + 1
        # if epochs < start_epoch:
        #     print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
        #           (opt.weights, ckpt['epoch'], epochs))
        #     epochs += ckpt['epoch']  # finetune additional epochs
        # else:
        #     epochs -= ckpt['epoch']  # finetune additional epochs

        # 混合精度训练参数是否导入
        if opt.amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        del ckpt

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf  定义学习率变化趋势
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  # 指定从哪个epoch开始

    #Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # model.yolo_layers = model.module.yolo_layers

    # dataset
    # 训练集的图像尺寸指定为multi_scale_range中最大的尺寸
    train_dataset = LoadImagesAndLabels(train_path, imgsz_train, batch_size,
                                        augment=True,
                                        hyp=hyp,  # augmentation hyperparameters
                                        rect=opt.rect,  # rectangular training
                                        cache_images=opt.cache_images,
                                        single_cls=opt.single_cls)

    # 验证集的图像尺寸指定为img_size(512)
    val_dataset = LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                      hyp=hyp,
                                      rect=True,  # 将每个batch的图像调整到合适大小，可减少运算量(并不是512x512标准尺寸)
                                      cache_images=opt.cache_images,
                                      single_cls=opt.single_cls)

    # dataloader
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=not opt.rect,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

    val_datasetloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=nw,
                                                    pin_memory=True,
                                                    collate_fn=val_dataset.collate_fn)

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    # 计算每个类别的目标个数，并计算每个类别的比重
    # model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device)  # attach class weights

    # start training
    # caching val_data when you have plenty of memory(RAM)
    # coco = None
    coco = get_coco_api_from_dataset(val_dataset)

    print("starting traning for %g epochs..." % epochs)
    print('Using %g dataloader workers' % nw)
    for epoch in range(start_epoch, epochs):
        mloss, lr = train_util.train_one_epoch(model, optimizer, train_dataloader,
                                               device, epoch,
                                               accumulate=accumulate,  # 迭代多少batch才训练完64张图片
                                               img_size=imgsz_train,  # 输入图像的大小
                                               multi_scale=multi_scale,
                                               grid_min=grid_min,  # grid的最小尺寸
                                               grid_max=grid_max,  # grid的最大尺寸
                                               gs=gs,  # grid step: 32
                                               print_freq=200,  # 每训练多少个step打印一次信息
                                               warmup=True,
                                               scaler=scaler)
        # update scheduler
        scheduler.step()

        if opt.notest is False or epoch == epochs - 1:
            # evaluate on the test dataset
            result_info = train_util.evaluate(model, val_datasetloader,
                                              coco=coco, device=device)

            coco_mAP = result_info[0]
            voc_mAP = result_info[1]
            coco_mAR = result_info[8]

            # write into tensorboard
            if tb_writer:
                tags = ['train/GIoU_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
                        "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

                # 这里mloss包含了4个部分：边界框损失、置信度损失、分类损失、总损失；
                for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                    tb_writer.add_scalar(tag, x, epoch)
                    # 以epoch为横坐标，x为竖坐标，tag为标题进行更新，

            # write into txt   coco的验证指标 保存在txt文件中

            with open(results_file, "a") as f:
                # 记录coco的12个指标加上训练总损失和lr
                result_info = [str(round(i, 4)) for i in result_info + [mloss.tolist()[-1]]] + [str(round(lr, 6))]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            # update best mAP(IoU=0.50:0.95)
            if coco_mAP > best_map:
                best_map = coco_mAP

            if opt.savebest is False:
                # save weights every epoch
                with open(results_file, 'r') as f:
                    save_files = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'training_results': f.read(),
                        'epoch': epoch,
                        'best_map': best_map}
                    if opt.amp:
                        save_files["scaler"] = scaler.state_dict()
                    torch.save(save_files, "./weights/yolov3spp-{}.pt".format(epoch))

            else:
                # only save best weights
                if best_map == coco_mAP:
                    with open(results_file, 'r') as f:
                        save_files = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map}
                        if opt.amp:
                            save_files["scaler"] = scaler.state_dict()
                        torch.save(save_files, best.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--cfg', type=str, default = 'cfg/dense35_bifpn.cfg', help="*.cfg path")  # cfg/preprocess_cfg/yolov3-sppf-bifpn.cfg
    parser.add_argument('--data', type=str, default = r'D:\BaiduNetdiskDownload\insulator\my_data\my_data.data', help = '*.data path')

    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')  # 超参数的路径
    parser.add_argument('--multi-scale', type=bool, default=True,
                        help='adjust (67%% - 150%%) img_size every 10 batches')  # 多尺度训练，默认开启
    parser.add_argument('--img-size', type=int, default=512, help='test size')  #512
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--savebest', type=bool, default=True, help='only save best checkpoint')   # 保存map最高的那次权重
    parser.add_argument('--notest', action='store_true', help='only test final epoch')  # 仅仅测试最后的epoch
    parser.add_argument('--cache-images', action='store_true', default= True,
                        help='cache images for faster training')  # 缓存图像以加快训练速度 default=True
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')  # default='weights/best_only3.pt'  yolov3spp-voc-512.pt
    # weights/yolov3spp-voc-512.pt  0225best_sppf_bifpn_conv
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', type=bool, default=False, help='Freeze non-output layers')
    # 设置为true会只训练预测器的三个卷积层  设置为false会训练除了Darknet53之后的所有权重
    # 先训练三个预测器再训练除darknet53之后的所有权重更好 video1:22min
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")
    opt = parser.parse_args()

    # 检查文件是否存在
    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.data)
    opt.hyp = check_file(opt.hyp)
    print(opt)

    with open(opt.hyp,'r', encoding='utf-8') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)


    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(comment=opt.name)
    torch.cuda.empty_cache()
    train(hyp)
    print("程序运行正常结束！")




