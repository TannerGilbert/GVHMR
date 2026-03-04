import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hmr4d.utils.geo.flip_utils import flip_heatmap_coco133
from hmr4d.utils.geo_transform import cvt_p2d_from_pm1_to_i
from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps

from .vitfeat_extractor import get_batch, get_batch_multiperson
from .vitpose_pytorch import build_model


class VitPoseWholebodyExtractor:
    def __init__(self, tqdm_leave=True, batch_size=16):
        ckpt_path = "inputs/checkpoints/vitpose/vitpose-h-coco-wholebody.pth"
        self.pose = build_model("ViTPose_huge_wholebody_256x192", ckpt_path)
        self.pose.cuda().eval()

        self.flip_test = True
        self.tqdm_leave = tqdm_leave
        self.batch_size = batch_size

    @torch.no_grad()
    def extract(self, video_path, bbx_xys, img_ds=0.5):
        # Get the batch
        if isinstance(video_path, str):
            if bbx_xys.ndim == 3:  # multiple persons (person_num, F, 3)
                imgs, bbx_xys = get_batch_multiperson(video_path, bbx_xys, img_ds=img_ds)
            else:
                imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        L, _, H, W = imgs.shape  # (L, 3, H, W)
        batch_size = self.batch_size
        vitpose = []
        for j in tqdm(range(0, L, batch_size), desc="ViTPose", leave=self.tqdm_leave):
            # Heat map
            imgs_batch = imgs[j : j + batch_size, :, :, 32:224].cuda()
            if self.flip_test:
                heatmap, heatmap_flipped = self.pose(torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)).chunk(2)
                heatmap_flipped = flip_heatmap_coco133(heatmap_flipped)
                heatmap = (heatmap + heatmap_flipped) * 0.5
                del heatmap_flipped
            else:
                heatmap = self.pose(imgs_batch.clone())  # (B, J, 64, 48)

            # postprocess from mmpose
            bbx_xys_batch = bbx_xys[j : j + batch_size]
            heatmap = heatmap.clone().cpu().numpy()
            center = bbx_xys_batch[:, :2].numpy()
            scale = (torch.cat((bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1) / 200).numpy()
            preds, maxvals = keypoints_from_heatmaps(heatmaps=heatmap, center=center, scale=scale, use_udp=True)
            kp2d = np.concatenate((preds, maxvals), axis=-1)
            kp2d = torch.from_numpy(kp2d)

            vitpose.append(kp2d.detach().cpu().clone())

        vitpose = torch.cat(vitpose, dim=0).clone()  # (F, 133, 3)
        return vitpose

    @torch.no_grad()
    def extract_multiperson(self, video_path, bbx_xys, img_ds=0.5):
        # Get the batch
        print(f"bbx_xys shape: {bbx_xys.shape}")
        if isinstance(video_path, str):
            # multiple persons (person_num, F, 3)
            if bbx_xys.ndim == 3:  # multiple persons (person_num, F, 3)
                imgs, bbx_xys = get_batch_multiperson(video_path, bbx_xys, img_ds=img_ds)
                person_num = bbx_xys.shape[0]
            else:
                imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
                person_num = 1
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        L, _, H, W = imgs.shape  # (L, 3, H, W)
        batch_size = self.batch_size
        vitpose = []
        for j in tqdm(range(0, L, batch_size), desc="ViTPose", leave=self.tqdm_leave):
            # Heat map
            imgs_batch = imgs[j : j + batch_size, :, :, 32:224]
            if self.flip_test:
                heatmap, heatmap_flipped = self.pose(torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)).chunk(2)
                heatmap_flipped = flip_heatmap_coco133(heatmap_flipped)
                heatmap = (heatmap + heatmap_flipped) * 0.5
                del heatmap_flipped
            else:
                heatmap = self.pose(imgs_batch.clone())  # (B, J, 64, 48)

            bbx_xys_batch = bbx_xys[j : j + batch_size]
            heatmap = heatmap.clone().cpu().numpy()
            center = bbx_xys_batch[:, :2].numpy()
            scale = (torch.cat((bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1) / 200).numpy()
            preds, maxvals = keypoints_from_heatmaps(heatmaps=heatmap, center=center, scale=scale, use_udp=True)
            kp2d = np.concatenate((preds, maxvals), axis=-1)
            kp2d = torch.from_numpy(kp2d)

            vitpose.append(kp2d)

        vitpose = torch.cat(vitpose, dim=0).clone().reshape(person_num, -1, 133, 3)  # (person_num, F, 133, 3)
        print(f"person_num: {person_num}, ViTPose wholebody shape: {vitpose.shape}")
        return vitpose, imgs  

