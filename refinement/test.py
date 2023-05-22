import logging
import os
import numpy as np
import random
from tqdm import tqdm
import yaml

import torch
import torch.utils.data

# datasets
from datasets.bcss import BCSSDataset
from datasets.camelyon import CamelyonDataset

# models
from models.linknet import LinkNet

# utils
from utils.metric import ConfusionMatrix


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    setup_seed(args.seed)

    # logger file and tensorboard dir
    final_log_file = os.path.join(args.dir, "logs", "test.txt")
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    device = torch.device('cuda')

    if args.dataset == "bcss":
        dataset = BCSSDataset(args.split_file, args.dataset_path, True,
                              prototype_mask_folder=args.prototype_mask_folder, return_gt=True)
        dataset_test = BCSSDataset(args.split_file, args.dataset_path, False,
                                   prototype_mask_folder=args.prototype_mask_folder, return_gt=True)
        num_classes = dataset.num_classes
    elif args.dataset == "camelyon":
        dataset = CamelyonDataset(args.split_file, args.dataset_path, True,
                                  prototype_mask_folder=args.prototype_mask_folder, return_gt=True)
        dataset_test = CamelyonDataset(args.split_file, args.dataset_path, False,
                                       prototype_mask_folder=args.prototype_mask_folder, return_gt=True)
        num_classes = dataset.num_classes
    else:
        raise NotImplementedError()

    logger.info("train set size:%d, test set size:%d" %
                (len(dataset), len(dataset_test)))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    if args.model == "linknet":
        model = LinkNet(num_classes)
    else:
        raise NotImplementedError()

    model.to(device)
    model_path = os.path.join(args.dir, "checkpoints", "best_checkpoint.pth")
    logger.info("resume from %s" % (model_path))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    # train
    model.eval()
    mat = ConfusionMatrix(num_classes=num_classes, device=device)
    with torch.no_grad():
        mat.reset()
        for image, target in tqdm(data_loader_test):
            image, target = image.to(device), target.to(device)
            output = model(image)
            mat.update(target.flatten(), output.argmax(1).flatten())
        logger.info('Validation Acc_class: {:.6f}({}), Validation Dice: {:.6f}({})'.format(mat.acc(
        ).mean().cpu().item(), mat.acc(), mat.dice().mean().cpu().item(), mat.dice().cpu().tolist()))
        logger.info(mat.mat)


def load_args_and_config(add_help=True):
    import argparse

    default_config_parser = parser = argparse.ArgumentParser(
        description='Training Config', add_help=add_help)
    parser.add_argument("--dir", type=str, help="dir")
    parser.add_argument("--dataset-name", type=str, help="dataset")

    # load from cmd
    given_configs, remaining = default_config_parser.parse_known_args()
    # load from config yaml
    with open(os.path.join(given_configs.dir, "code", "configs", given_configs.dataset_name+".yaml")) as f:
        cfg = yaml.safe_load(f)
        default_config_parser.set_defaults(**cfg)
    args = default_config_parser.parse_args()

    return args


if __name__ == "__main__":
    args = load_args_and_config()
    main(args)
