import datetime
import logging
import os
import time
import numpy as np
import random
import yaml

import torch
import torch.utils.data
from torch import nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# datasets
from datasets.bcss import BCSSDataset
from datasets.camelyon import CamelyonDataset

# models
from models.linknet import LinkNet

# utils
from utils.metric import ConfusionMatrix
from utils.util import create_logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def main(args):
    setup_seed(args.seed)

    if args.local_rank == 0:
        logger, ckpt_output_dir, tb_log_dir = create_logger(args)
        writer = SummaryWriter(tb_log_dir)
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        logger.info(args_text)

    if args.gpus > 1:
        args.distributed = True
        torch.cuda.set_device(args.local_rank)
        args.dist_backend = "nccl"
        logging.info(
            f"| distributed init (rank {args.local_rank}): {args.dist_url}")
        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url)

    world_size = get_world_size()
    device = torch.device(args.device, args.local_rank)

    if args.dataset == "bcss":
        dataset = BCSSDataset(args.split_file, args.dataset_path, True,
                              prototype_mask_folder=args.prototype_mask_folder, return_gt=args.fully_supervised)
        dataset_test = BCSSDataset(args.split_file, args.dataset_path, False,
                                   prototype_mask_folder=args.prototype_mask_folder, return_gt=True)
        num_classes = dataset.num_classes
    elif args.dataset == "camelyon":
        dataset = CamelyonDataset(args.split_file, args.dataset_path, True,
                                  prototype_mask_folder=args.prototype_mask_folder, return_gt=args.fully_supervised)
        dataset_test = CamelyonDataset(args.split_file, args.dataset_path, False,
                                       prototype_mask_folder=args.prototype_mask_folder, return_gt=True)
        num_classes = dataset.num_classes
    else:
        raise NotImplementedError()

    if args.local_rank == 0:
        logger.info("train set size:%d, test set size:%d" %
                    (len(dataset), len(dataset_test)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed + worker_id)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=train_sampler,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, num_workers=args.workers, sampler=test_sampler, worker_init_fn=worker_init_fn
    )

    if args.model == "linknet":
        model = LinkNet(num_classes, args.pretrain)
    else:
        raise NotImplementedError()

    model.to(device)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    iters_per_epoch = len(data_loader)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda iter: 1)
    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iter: args.lr_warmup_decay + (
                1 - args.lr_warmup_decay) * iter / warmup_iters if iter < warmup_iters else (1 - (iter - warmup_iters) / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))) ** 0.9)
        elif args.lr_warmup_method == "constant":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iter: args.lr_warmup_decay if iter < warmup_iters else (
                1 - (iter - warmup_iters) / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))) ** 0.9)
        else:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda x: (
                    1 - x / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))) ** 0.9
            )

    if args.resume:
        if args.local_rank == 0:
            logger.info("resume from %s" % (args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        args.start_epoch = checkpoint["epoch"]
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank])

    # train
    start_time = time.time()
    iter_num, iter_total = iters_per_epoch * \
        args.start_epoch, iters_per_epoch * args.epochs
    best_metric = 0.0
    args.save_freq = iters_per_epoch if args.save_freq == 0 else args.save_freq
    args.check_freq = iters_per_epoch if args.check_freq == 0 else args.check_freq

    ce_loss_func = nn.CrossEntropyLoss(reduction="none", ignore_index=255)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        for i, data in enumerate(data_loader):
            iter_num += 1
            image, target = data
            image, target = image.to(device), target.to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(image)
                loss = ce_loss_func(output, target)
                loss = loss.mean()

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()

            reduced_loss = reduce_tensor(loss) / world_size
            if args.local_rank == 0 and iter_num % args.print_freq == 0:
                logger.info(
                    'Epoch [ %d | %d ] Iteration [ %d | %d ] iter [ %d | %d ] lr : %.4f, loss_ce : %.4f' %
                    (epoch, args.epochs, iter_num, iter_total, i, iters_per_epoch, optimizer.param_groups[0]["lr"], reduced_loss.item()))
                writer.add_scalar('train/loss', reduced_loss.item(), iter_num)
                writer.add_scalar(
                    'train/lr', optimizer.param_groups[0]["lr"], iter_num)


        # eval
        checkpoint = {
            "model": model.module.state_dict() if args.gpus > 1 else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()

        model.eval()
        mat = ConfusionMatrix(num_classes=num_classes, device=device)
        with torch.no_grad():
            for i, data in enumerate(data_loader_test):
                image, target = data
                image, target = image.to(device), target.to(device)
                output = model(image)
                original_num = len(data_loader_test.dataset) % (
                    args.batch_size * world_size)
                if i == len(data_loader_test) - 1 and args.distributed and original_num > 0:
                    # deal with pad in ddp
                    base_num = original_num // world_size
                    res_num = original_num % world_size
                    idx = base_num + 1 if args.local_rank < res_num else base_num
                    mat.update(target[:idx].flatten(),
                               output[:idx].argmax(1).flatten())
                else:
                    mat.update(target.flatten(), output.argmax(1).flatten())

        torch.distributed.barrier()
        torch.distributed.all_reduce(mat.mat)

        if args.local_rank == 0:
            dice = mat.dice()
            metric = dice.mean()
            logger.info('Test acc {}({}) dice: {:.6f} \n class dice: {}'.format(
                mat.acc().mean(), mat.acc(), metric, [f"{i:.1f}" for i in (dice * 100).tolist()]))

            for i in range(dice.shape[0]):
                writer.add_scalar('val/dice%d' % i, dice[i], iter_num)
            writer.add_scalar('val/mean_dice', metric, iter_num)
            writer.add_scalar('val/pixel_acc', mat.acc().mean(), iter_num)

            if metric > best_metric:
                best_metric = metric
                torch.save(checkpoint, os.path.join(
                    ckpt_output_dir, "best_checkpoint.pth"))
                logger.info('save best model...')

            torch.save(checkpoint, os.path.join(ckpt_output_dir,
                       "epoch{}_checkpoint.pth".format(epoch)))
        model.train()

    if args.local_rank == 0:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training time {total_time_str}")


def load_args_and_config(add_help=True):
    import argparse

    default_config_parser = parser = argparse.ArgumentParser(
        description='Training Config', add_help=add_help)
    parser.add_argument("--config", type=str, help="experiement config")
    parser.add_argument("--distributed", default=False,
                        type=bool, help="use distributed")
    parser.add_argument("--dist-backend", default="nccl",
                        type=str, help="distributed backend")
    parser.add_argument("--world-size", default=1, type=int,
                        help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--local_rank", default=-1, type=int)

    # load from cmd
    given_configs, remaining = default_config_parser.parse_known_args()
    # load from config yaml
    with open(given_configs.config) as f:
        cfg = yaml.safe_load(f)
        default_config_parser.set_defaults(**cfg)
    args = default_config_parser.parse_args()

    # change arg type
    args.lr = float(args.lr)
    args.weight_decay = float(args.weight_decay)
    return args


if __name__ == "__main__":
    args = load_args_and_config()
    main(args)
