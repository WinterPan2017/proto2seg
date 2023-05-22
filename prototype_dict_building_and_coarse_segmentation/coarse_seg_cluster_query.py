# 根据组织原型点进行分割
import yaml
import os
import logging
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import cv2

from modules import resnet, network, transform
from datasets.prototype_seg import PrototypeSegDataset, BCSS_Seg
from evaluation import metric

from sklearn.cluster import KMeans

def cluster_query(feature_map, centroid, mapkey, n_cluster, device):
    """suport batch size = 1"""
    _, c, h, w = feature_map.size()
    # cluster and query
    embeddings = feature_map.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, c)
    estimator = KMeans(n_clusters=n_cluster, init='k-means++')
    estimator.fit(embeddings)
    cluster_pred = estimator.labels_
    cluster_centroid = estimator.cluster_centers_
    cluster_centroid = torch.from_numpy(
        cluster_centroid).float().to(device=device)
    sim = (cluster_centroid @ centroid.T) / (torch.linalg.norm(cluster_centroid,
                                                               dim=1, keepdim=True) @ torch.linalg.norm(centroid, dim=1, keepdim=True).T)
    query_label = mapkey[torch.argmax(sim, dim=1, keepdim=True)].float()

    pred_mask = torch.zeros((h*w, )).float().to(device=device)
    for i in np.unique(cluster_pred):
        pred_mask[cluster_pred == i] = query_label[i]
    pred = pred_mask.reshape(1, 1, h, w)

    cluster_pred = torch.from_numpy(
        cluster_pred.reshape(1, 1, h, w)).int().to(device=device)

    return pred, cluster_pred


def direct_query(feature_map, centroid, mapkey, device):
    batch_size, c, h, w = feature_map.size()
    fm = feature_map.view(batch_size, c, -1).permute(0,
                                                     2, 1)  # (1, 64x64, 512)
    sim = (fm @ centroid.T) / (torch.linalg.norm(fm, dim=2, keepdim=True)
                               @ torch.linalg.norm(centroid, dim=1, keepdim=True).T)
    pred = mapkey[torch.argmax(sim, dim=2, keepdim=True)].view(
        batch_size, h, w, 1).permute(0, 3, 1, 2)
    return pred


if __name__ == "__main__":
    default_config_parser = parser = argparse.ArgumentParser(
        description='Config', add_help=False)
    parser.add_argument("--config", type=str, help="config file")
    parser.add_argument("--folder", default=-1, type=int)
    parser.add_argument("--n", default=5, type=int)

    # load from cmd
    given_configs, remaining = default_config_parser.parse_known_args()
    # load from config yaml
    with open(given_configs.config) as f:
        cfg = yaml.safe_load(f)
        default_config_parser.set_defaults(**cfg)
    args = default_config_parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # store
    exp_name = args.dataset + "_" + \
        str(args.n_cluster) + "_" + args.cluster_to_tissue_method
    if args.folder != -1:
        output_dir = os.path.join("./output", exp_name, str(args.folder))
    else:
        output_dir = os.path.join(
            "./output", exp_name, str(len(os.listdir(os.path.join("./output", exp_name)))-1))

    # logger
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
        train_dataset = PrototypeSegDataset(
            df_list=args.seg_df_list,
            datadir=args.data_dir,
            train=True
        )
        val_dataset = PrototypeSegDataset(
            df_list=args.seg_df_list,
            datadir=args.data_dir,
            train=False
        )
        class_num = 2
    elif args.dataset == "bcss":
        train_dataset = BCSS_Seg(
            df_list=args.seg_df_list,
            datadir=args.data_dir,
            train=True
        )
        val_dataset = BCSS_Seg(
            df_list=args.seg_df_list,
            datadir=args.data_dir,
            train=False
        )
        class_num = 5
    else:
        raise NotImplementedError
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=6,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=6,
    )

    # load model
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model.load_state_dict(torch.load(
        args.model_path, map_location=device.type)['net'])
    model.eval()
    backbone = model.resnet
    model.to(device)
    backbone.to(device)

    # coarase seg
    centroid = np.load(os.path.join(output_dir, "y_centroid.npy"))  # (K, 512)
    mapkey = np.load(os.path.join(output_dir, "mapkeymat.npy"))  # (K, )

    delete_idx = []
    if len(delete_idx) > 0:
        logger.info('delete clusters index: {} '.format(delete_idx))
        centroid = np.delete(centroid, delete_idx, 0)
        mapkey = np.delete(mapkey, delete_idx)

    centroid = torch.from_numpy(centroid).to(device)
    mapkey = torch.from_numpy(mapkey).to(device)

    train_metric = metric.Evaluator(class_num)
    val_metric = metric.Evaluator(class_num)
    all_metric = metric.Evaluator(class_num)

    train_metric_cq = metric.Evaluator(class_num)
    val_metric_cq = metric.Evaluator(class_num)
    all_metric_cq = metric.Evaluator(class_num)

    if args.save_prototype_seg:
        store = []
        dq_save_path = os.path.join(
            args.save_prototype_seg_dir, exp_name+"_0.8_DQ")
        save_path = os.path.join(
            args.save_prototype_seg_dir, exp_name+"_0.8_CQ{}".format(args.n))
        save_vis_path = os.path.join(
            args.save_prototype_seg_dir, exp_name+"_0.8_GT_DQ_CQ_vis".format(args.n))

        os.makedirs(dq_save_path, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_vis_path, exist_ok=True)

    with torch.no_grad():
        for i, (datas, msks, labels, names) in enumerate(tqdm(train_data_loader)):
            datas = datas.to(device)
            fm = backbone.get_fm(datas)  # (1, 512, 64, 64)
            fm = F.normalize(fm, dim=1)
            batch_size, c, h, w = fm.size()

            d_pred = direct_query(fm, centroid, mapkey, device)
            cq_pred, cq_cluster_pred = cluster_query(fm, centroid, mapkey, len(
                np.unique(d_pred.cpu().numpy()))*args.n, device)

            cq_pred = F.interpolate(cq_pred.float(
            ), (args.seg_image_size, args.seg_image_size), mode="nearest").int()
            cq_cluster_pred = F.interpolate(cq_cluster_pred.float(
            ), (args.seg_image_size, args.seg_image_size), mode="nearest").int()
            d_pred = F.interpolate(
                d_pred.float(), (args.seg_image_size, args.seg_image_size), mode="nearest").int()
            train_metric.add_batch(msks.cpu().detach().numpy(
            ).ravel(), d_pred.cpu().detach().numpy().ravel())
            train_metric_cq.add_batch(msks.cpu().detach().numpy(
            ).ravel(), cq_pred.cpu().detach().numpy().ravel())
            if args.save_prototype_seg:
                for j, name in enumerate(names):
                    cv2.imwrite(os.path.join(save_path, name),
                                cq_pred.cpu().detach().numpy()[j][0])
                    cv2.imwrite(os.path.join(dq_save_path, name),
                                d_pred.cpu().detach().numpy()[j][0])
                    store.append(name)
                    print(len(store), len(np.unique(np.array(store))))

        for i, (datas, msks, labels, names) in enumerate(tqdm(val_data_loader)):
            datas = datas.to(device)
            fm = backbone.get_fm(datas)  # (1, 512, 64, 64)
            fm = F.normalize(fm, dim=1)
            batch_size, c, h, w = fm.size()

            d_pred = direct_query(fm, centroid, mapkey, device)
            cq_pred, cq_cluster_pred = cluster_query(fm, centroid, mapkey, len(
                np.unique(d_pred.cpu().numpy()))*args.n, device)
            cq_pred = F.interpolate(cq_pred.float(
            ), (args.seg_image_size, args.seg_image_size), mode="nearest").int()  # up-sample to full size
            cq_cluster_pred = F.interpolate(cq_cluster_pred.float(
            ), (args.seg_image_size, args.seg_image_size), mode="nearest").int()
            d_pred = F.interpolate(
                d_pred.float(), (args.seg_image_size, args.seg_image_size), mode="nearest").int()
            val_metric.add_batch(msks.cpu().detach().numpy(
            ).ravel(), d_pred.cpu().detach().numpy().ravel())
            val_metric_cq.add_batch(msks.cpu().detach().numpy(
            ).ravel(), cq_pred.cpu().detach().numpy().ravel())
            if args.save_prototype_seg:
                for j, name in enumerate(names):
                    cv2.imwrite(os.path.join(save_path, name),
                                cq_pred.cpu().detach().numpy()[j][0])
                    cv2.imwrite(os.path.join(dq_save_path, name),
                                d_pred.cpu().detach().numpy()[j][0])
                    img = datas[j].cpu().numpy().transpose(1, 2, 0)*255
                    if args.dataset == "bcss":
                        gt = cv2.applyColorMap(
                            (msks[j]*63).cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
                        d_heatmap = cv2.applyColorMap(
                            (d_pred[j][0]*63).cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
                        cq_heatmap = cv2.applyColorMap(
                            (cq_pred[j][0]*63).cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
                    else:
                        gt = cv2.applyColorMap(
                            (msks[j]*255).cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
                        d_heatmap = cv2.applyColorMap(
                            (d_pred[j][0]*255).cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
                        cq_heatmap = cv2.applyColorMap(
                            (cq_pred[j][0]*255).cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
                    
                    pad = np.zeros((img.shape[0], 3, 3))
                    vis = np.concatenate(
                        (img, pad, gt, pad, d_heatmap, pad, cq_heatmap), axis=1)
                    cv2.imwrite(os.path.join(save_vis_path, name), vis)
                    store.append(name)
                    print(len(store), len(np.unique(np.array(store))))


    logger.info("cluster num: {}".format(args.n))
    logger.info('direct query Train Acc: {:.6f}, Train Acc_class: {:.6f}, Train Dice: {:.6f}({})(Foreground Mean Dice: {}), Train mIOU: {:.6f}'.format(train_metric.Pixel_Accuracy(
    ), train_metric.Pixel_Accuracy_Class().mean(), train_metric.Pixel_Dice().mean(), train_metric.Pixel_Dice(), train_metric.Pixel_Dice()[1:].mean(), train_metric.Mean_Intersection_over_Union()))
    logger.info('Validation Acc: {:.6f}, Validation Acc_class: {:.6f}{}, Validation Dice: {:.6f}({})(Foreground Mean Dice: {}), Validation mIOU: {:.6f} '.format(val_metric.Pixel_Accuracy(
    ), val_metric.Pixel_Accuracy_Class().mean(), val_metric.Pixel_Accuracy_Class(), val_metric.Pixel_Dice().mean(), val_metric.Pixel_Dice(), val_metric.Pixel_Dice()[1:].mean(), val_metric.Mean_Intersection_over_Union()))
    logger.info('cluster query Train Acc: {:.6f}, Train Acc_class: {:.6f}, Train Dice: {:.6f}({})(Foreground Mean Dice: {}), Train mIOU: {:.6f}'.format(train_metric_cq.Pixel_Accuracy(
    ), train_metric_cq.Pixel_Accuracy_Class().mean(), train_metric_cq.Pixel_Dice().mean(), train_metric_cq.Pixel_Dice(), train_metric_cq.Pixel_Dice()[1:].mean(), train_metric_cq.Mean_Intersection_over_Union()))
    logger.info('Validation Acc: {:.6f}, Validation Acc_class: {:.6f}{}, Validation Dice: {:.6f}({})(Foreground Mean Dice: {}), Validation mIOU: {:.6f} '.format(val_metric_cq.Pixel_Accuracy(), val_metric_cq.Pixel_Accuracy_Class(
    ).mean(), val_metric_cq.Pixel_Accuracy_Class(), val_metric_cq.Pixel_Dice().mean(), val_metric_cq.Pixel_Dice(), val_metric_cq.Pixel_Dice()[1:].mean(), val_metric_cq.Mean_Intersection_over_Union()))
