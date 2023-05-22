
from evaluation import evaluation
from modules import resnet, network, transform
from datasets.cam import Cam16
from datasets.bcss import BCSS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import copy
from torch.utils import data
import numpy as np
import torch.nn.functional as F
import torchvision
import torch
import argparse
import yaml
import os
import logging
import shutil
import warnings
warnings.filterwarnings('ignore')


def get_patch_embeddding(dataloader, output_dir, args):
    """get patch embedding"""
    feature_path = os.path.join(
        output_dir, "../../feature_" + args.dataset + ".npy")
    label_path = os.path.join(
        output_dir, "../../label_" + args.dataset + ".npy")
    if os.path.exists(feature_path) and os.path.exists(label_path):
        features = np.load(feature_path)
        labels = np.load(label_path)
    else:
        # load model
        res = resnet.get_resnet(args.resnet)
        model_ = network.Network(res, args.feature_dim, class_num)
        model_.load_state_dict(torch.load(
            args.model_path, map_location=device.type)['net'])
        model_.eval()
        backbone = model.resnet
        model_.to(device)
        backbone.to(device)
        model = backbone
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(dataloader)):
                batch_size = x.size(0)
                x = x.to(device)
                x = model(x)
                x = F.normalize(x, dim=1)
                x = x.cpu().detach().numpy().reshape(batch_size, -1)
                if i == 0:
                    featureList = x
                    labelsList = y
                else:
                    featureList = np.append(featureList, x, axis=0)
                    labelsList = np.append(labelsList, y, axis=0)
        features = np.array(featureList)
        labels = np.array(labelsList)
        np.save(feature_path, features)
        np.save(label_path, labels)
    return features, labels


def cluster(featureList, labelsList, n_clusters, cluster_to_tissue_method, prototype_threshold, df, output_dir):
    # cluster
    estimator = KMeans(n_clusters=n_clusters, init='k-means++')

    estimator.fit(featureList)
    y_pred = estimator.labels_
    y_centroid = estimator.cluster_centers_
    gt_unique, counts = np.unique(np.array(y_pred), return_counts=True)
    f_pred = y_pred.copy()

    # map tissuse method
    sample_method, sample_num = cluster_to_tissue_method.split(":")
    sample_num = -1 if sample_num == "all" else int(sample_num)

    sim = (y_centroid @ featureList.T) / (np.linalg.norm(y_centroid, axis=1,
                                                         keepdims=True) @ np.linalg.norm(featureList, axis=1, keepdims=True).T)

    maplist = np.zeros(n_clusters)
    delete_idx = []
    for key in gt_unique:
        loc = np.where(y_pred == key)

        if sample_method == "top":
            sorted_idx = np.argsort(sim[:, loc[0]])[:, ::-1]
            sample_loc = loc[0][sorted_idx[key, :sample_num]]
        elif sample_method == "sample":
            interval = loc[0].shape[0] // sample_num if sample_num != -1 else 1
            interval = 1 if interval == 0 else interval
            logger.info("total num:{},  cluster num:{}, cluster sample num:{}, interval:{}".format(
                featureList.shape[0], loc[0].shape[0], sample_num, interval))
            sorted_idx = np.argsort(sim[:, loc[0]])[:, ::-1]
            sample_loc = loc[0][sorted_idx[key, ::interval]][:sample_num]
        else:
            raise NotImplementedError

        pd = labelsList[sample_loc].copy()
        tumor_rate = [(pd == i).sum() / len(pd)
                      for i in range(len(np.unique(labelsList)))]
        maxlabel = np.argmax(np.bincount(pd))
        f_pred[loc] = maxlabel
        if np.array(tumor_rate).max() < prototype_threshold:
            logger.info("Map fine cluster " + str(key)+" to tissue " +
                        str(maxlabel)+" with tumor rate " + str(tumor_rate) + " delete")
            delete_idx.append(key)
            maplist[key] = maxlabel
        else:
            logger.info("Map fine cluster " + str(key)+" to tissue " +
                        str(maxlabel)+" with tumor rate " + str(tumor_rate))
            maplist[key] = maxlabel
    if len(delete_idx) > 0:
        y_centroid = np.delete(y_centroid, delete_idx, 0)
        maplist = np.delete(maplist, delete_idx)
    np.save(os.path.join(output_dir, "mapkeymat.npy"), np.array(maplist))
    nmi, ari, f, acc = evaluation.evaluate(
        np.array(labelsList).ravel(), np.array(f_pred).ravel())
    logger.info('Map cluster: NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(
        nmi, ari, f, acc))


    finelabmat = np.array(y_pred)
    logger.info("finelabmat shape: %s" % (finelabmat.shape, ))
    np.save(os.path.join(output_dir, 'y_pred.npy'), finelabmat)
    logger.info("y_centroid shape: %s" % (y_centroid.shape,))
    np.save(os.path.join(output_dir, 'y_centroid.npy'), y_centroid)
    np.save(os.path.join(output_dir, 'y_fpred.npy'), np.array(f_pred))
    logger.info('Fine cluster: NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(
        nmi, ari, f, acc))

    return y_pred


def vis(featureList, labelsList, y_pred, output_dir):
    x_encode = TSNE(n_components=2, random_state=66).fit_transform(featureList)
    gt_unique, counts = np.unique(np.array(labelsList), return_counts=True)
    labels_freq = dict(zip(gt_unique, counts))
    logger.info("number of gt clusters : %d" % len(gt_unique))
    logger.info(labels_freq)
    cmap = plt.get_cmap('viridis', len(gt_unique))
    v_x = x_encode
    v_y = labelsList
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)
    classes = gt_unique
    lmap = {0: "Others", 1: "Tum.", 2: "Stm.", 3: "Inf.", 4: "Nec."}
    for key in classes:
        ix = np.where(v_y == key)
        ax.scatter(v_x[ix][:, 0], v_x[ix][:, 1],
                   color=cmap(key), label=lmap[key])
    ax.legend()
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, 'simclr-camgt-tsne.png'))

    predsList = y_pred
    gt_unique, counts = np.unique(np.array(predsList), return_counts=True)
    labels_freq = dict(zip(gt_unique, counts))
    logger.info("number of pred clusters : %d" % len(gt_unique))
    logger.info(labels_freq)
    cmap = plt.get_cmap('plasma', len(gt_unique))
    v_x = x_encode
    v_y = predsList
    # fig = plt.figure(figsize=(14,8))
    fig = plt.figure(figsize=(28, 16))
    ax = fig.add_subplot(1, 1, 1)
    classes = gt_unique
    for key in classes:
        ix = np.where(v_y == key)
        ax.scatter(v_x[ix[0]][:, 0], v_x[ix[0]]
                   [:, 1], color=cmap(key), label=key)
        ax.text(np.mean(v_x[ix[0]][:, 0]), np.mean(
            v_x[ix[0]][:, 1]), key, fontsize=18, bbox=dict(facecolor='white', alpha=0.5))
    # ax.legend()
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, 'simclr-camkmeans-tsne.png'))


if __name__ == "__main__":

    default_config_parser = parser = argparse.ArgumentParser(
        description='Config', add_help=False)
    parser.add_argument("--config", type=str, help="config file")
    # load from cmd
    given_configs, remaining = default_config_parser.parse_known_args()
    # load from config yaml
    with open(given_configs.config) as f:
        cfg = yaml.safe_load(f)
        default_config_parser.set_defaults(**cfg)
    args = default_config_parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # store config
    exp_name = args.dataset + "_" + \
        str(args.n_cluster) + "_" + args.cluster_to_tissue_method
    output_dir = os.path.join("./output", exp_name)
    num = len(os.listdir(output_dir)) if os.path.exists(output_dir) else 0
    output_dir = os.path.join(output_dir, str(num))
    os.makedirs(output_dir, exist_ok=False)

    # copy config
    shutil.copyfile(args.config, os.path.join(
        output_dir, args.config.split("/")[-1]))

    # logger setting
    logging.basicConfig(filename=os.path.join(
        output_dir, "log.txt"), format="%(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # fixed random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load dataset
    if args.dataset == "cam":
        dataset = Cam16(
            df_list=args.df_list,
            train=False,
            transform=transform.Transforms(
                size=args.image_size).test_transform,
        )
        class_num = 2
        df = dataset.list_df
    elif args.dataset == "bcss":
        dataset = BCSS(
            df_list=args.df_list,
            train=False,
            transform=transform.Transforms(
                size=args.image_size).test_transform,
        )
        class_num = 5
        df = dataset.list_df
        df = df.rename(columns={'patch_path': 'img', 'mask_path': 'mask'})
    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # patch embedding
    featureList, labelsList = get_patch_embeddding(
        data_loader, output_dir, args)
    logger.info("featureList shape: %s,labelsList shape : %s" %
                (featureList.shape, labelsList.shape))

    # cluster
    y_pred = cluster(featureList, labelsList, args.n_cluster,
                     args.cluster_to_tissue_method, args.prototype_threshold, df, output_dir)

    # vis
    vis(featureList, labelsList, y_pred, output_dir)
