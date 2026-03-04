import argparse
import copy
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from einops import einsum, rearrange
from hmr4d.configs import register_store_gvhmr
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.geo.hmr_cam import (convert_K_to_K4, create_camera_sensor,
                                     estimate_K, get_bbx_xys_from_xyxy)
from hmr4d.utils.geo_transform import (apply_T_on_points, compute_cam_angvel,
                                       compute_T_ayfz2ay)
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.preproc import Extractor, SimpleVO, Tracker, VitPoseExtractor
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import (get_video_lwh, get_video_reader,
                                        get_writer, merge_videos_horizontal,
                                        read_video_np, save_video)
from hmr4d.utils.vis.cv2_utils import (draw_bbx_xyxy_on_image_batch,
                                       draw_coco17_skeleton_batch)
from hmr4d.utils.vis.renderer import (Renderer, get_global_cameras_static,
                                      get_ground_params_from_points)
from hydra import compose, initialize_config_module
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  quaternion_to_matrix)
from tqdm import tqdm

CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--use_dpvo", action="store_true", help="If true, use DPVO. By default not using DPVO.")
    parser.add_argument(
        "--f_mm",
        type=int,
        default=None,
        help="Focal length of fullframe camera in mm. Leave it as None to use default values."
        "For iPhone 15p, the [0.5x, 1x, 2x, 3x] lens have typical values [13, 24, 48, 77]."
        "If the camera zoom in a lot, you can try 135, 200 or even larger values.",
    )
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    parser.add_argument(
        "--use_hamer_hands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, run HaMeR and fuse hand pose into SMPL-X output.",
    )
    parser.add_argument(
        "--hamer_checkpoint",
        type=str,
        default=None,
        help="Path to HaMeR checkpoint. If None, use HaMeR default checkpoint.",
    )
    parser.add_argument("--hamer_batch_size", type=int, default=8, help="Batch size for HaMeR inference.")
    parser.add_argument(
        "--hamer_rescale_factor",
        type=float,
        default=2.0,
        help="BBox padding scale factor for HaMeR hand crops.",
    )
    args = parser.parse_args()

    # Input
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")
    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"use_dpvo={args.use_dpvo}",
        ]
        if args.f_mm is not None:
            overrides.append(f"f_mm={args.f_mm}")

        # Allow to change output root
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=CRF)
        for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc=f"Copy"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    return cfg, args


def _load_hamer_modules():
    try:
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.models import DEFAULT_CHECKPOINT as DEFAULT_CHECKPOINT_HAMER
        from hamer.models import load_hamer
        from hamer.utils import recursive_to
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[3]
        hamer_root = repo_root / "hand_reconstruction" / "hamer"
        if not hamer_root.exists():
            raise
        if str(hamer_root) not in sys.path:
            sys.path.insert(0, str(hamer_root))
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.models import DEFAULT_CHECKPOINT as DEFAULT_CHECKPOINT_HAMER
        from hamer.models import load_hamer
        from hamer.utils import recursive_to
    return ViTDetDataset, load_hamer, DEFAULT_CHECKPOINT_HAMER, recursive_to


def _load_vitpose_wholebody(device):
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                             process_mmdet_results)

    repo_root = Path(__file__).resolve().parents[3]
    hamer_root = repo_root / "hand_reconstruction" / "hamer"
    cfg_path = (
        hamer_root
        / "third-party"
        / "ViTPose"
        / "configs"
        / "wholebody"
        / "2d_kpt_sview_rgb_img"
        / "topdown_heatmap"
        / "coco-wholebody"
        / "ViTPose_huge_wholebody_256x192.py"
    )
    ckpt_path = hamer_root / "_DATA" / "vitpose_ckpts" / "vitpose+_huge" / "wholebody.pth"
    assert cfg_path.exists(), f"ViTPose config not found: {cfg_path}"
    assert ckpt_path.exists(), f"ViTPose checkpoint not found: {ckpt_path}"
    model = init_pose_model(str(cfg_path), str(ckpt_path), device=device)

    def predict_pose_rgb(img_rgb, det_xyxy_score):
        img_bgr = img_rgb[:, :, ::-1]
        person_results = process_mmdet_results([det_xyxy_score], 1)
        out, _ = inference_top_down_pose_model(
            model,
            img_bgr,
            person_results=person_results,
            bbox_thr=0.5,
            format="xyxy",
        )
        return out

    return predict_pose_rgb


@torch.no_grad()
def run_hamer_preprocess(cfg, args):
    if not args.use_hamer_hands:
        Log.info("[Preprocess] Skip HaMeR hand stage.")
        return

    paths = cfg.paths
    mano_params_path = Path(paths.get("mano_params", Path(cfg.preprocess_dir) / "mano_params.pt"))
    if mano_params_path.exists():
        Log.info(f"[Preprocess] mano_params from {mano_params_path}")
        return

    assert torch.cuda.is_available(), "HaMeR stage requires CUDA."

    ViTDetDataset, load_hamer, default_hamer_ckpt, recursive_to = _load_hamer_modules()
    hamer_ckpt = args.hamer_checkpoint if args.hamer_checkpoint is not None else default_hamer_ckpt
    hamer_model, model_cfg_hamer = load_hamer(hamer_ckpt)
    hamer_model = hamer_model.cuda().eval()
    vitpose_wholebody = _load_vitpose_wholebody("cuda")

    bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"].cpu().numpy()
    reader = get_video_reader(cfg.video_path)
    length = get_video_lwh(cfg.video_path)[0]

    eye3 = torch.eye(3, dtype=torch.float32)
    left_go_list, left_hp_list, left_valid_list = [], [], []
    right_go_list, right_hp_list, right_valid_list = [], [], []

    for i, frame in tqdm(enumerate(reader), total=length, desc="HaMeR"):
        frame_rgb = frame
        frame_bgr = frame_rgb[:, :, ::-1]
        x1, y1, x2, y2 = bbx_xyxy[i].astype(np.float32)
        det = np.array([[x1, y1, x2, y2, 1.0]], dtype=np.float32)
        vitposes_out = vitpose_wholebody(frame_rgb, det)

        bboxes = []
        sides = []
        for vitpose in vitposes_out:
            left_hand_keyp = vitpose["keypoints"][-42:-21]
            right_hand_keyp = vitpose["keypoints"][-21:]

            valid = left_hand_keyp[:, 2] > 0.5
            if int(valid.sum()) > 3:
                keyp = left_hand_keyp[valid]
                bboxes.append([keyp[:, 0].min(), keyp[:, 1].min(), keyp[:, 0].max(), keyp[:, 1].max()])
                sides.append(0)
            valid = right_hand_keyp[:, 2] > 0.5
            if int(valid.sum()) > 3:
                keyp = right_hand_keyp[valid]
                bboxes.append([keyp[:, 0].min(), keyp[:, 1].min(), keyp[:, 0].max(), keyp[:, 1].max()])
                sides.append(1)

        left_go = eye3.clone()
        left_hp = eye3.clone()[None].repeat(15, 1, 1)
        right_go = eye3.clone()
        right_hp = eye3.clone()[None].repeat(15, 1, 1)
        left_valid = False
        right_valid = False

        if len(bboxes) > 0:
            dataset = ViTDetDataset(
                model_cfg_hamer,
                frame_bgr,
                np.asarray(bboxes, dtype=np.float32),
                np.asarray(sides, dtype=np.float32),
                rescale_factor=args.hamer_rescale_factor,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.hamer_batch_size,
                shuffle=False,
                num_workers=0,
            )
            for batch in dataloader:
                batch = recursive_to(batch, "cuda")
                out = hamer_model(batch)
                go_batch = out["pred_mano_params"]["global_orient"].detach().cpu()[:, 0]
                hp_batch = out["pred_mano_params"]["hand_pose"].detach().cpu()
                side_batch = batch["right"].detach().cpu().numpy().astype(np.int32)

                for n, side in enumerate(side_batch):
                    if side == 0 and not left_valid:
                        left_go = go_batch[n]
                        left_hp = hp_batch[n]
                        left_valid = True
                    elif side == 1 and not right_valid:
                        right_go = go_batch[n]
                        right_hp = hp_batch[n]
                        right_valid = True

        left_go_list.append(left_go)
        left_hp_list.append(left_hp)
        left_valid_list.append(left_valid)
        right_go_list.append(right_go)
        right_hp_list.append(right_hp)
        right_valid_list.append(right_valid)

    reader.close()

    all_mano_params = {
        "left_hand_global_orient": torch.stack(left_go_list, dim=0),
        "left_hand_pose": torch.stack(left_hp_list, dim=0),
        "left_hand_valid": torch.tensor(left_valid_list, dtype=torch.bool),
        "right_hand_global_orient": torch.stack(right_go_list, dim=0),
        "right_hand_pose": torch.stack(right_hp_list, dim=0),
        "right_hand_valid": torch.tensor(right_valid_list, dtype=torch.bool),
    }
    torch.save(all_mano_params, mano_params_path)
    Log.info(f"[Preprocess] Saved mano_params to {mano_params_path}")


def _compute_global_rotmats(global_orient_mat, body_pose_mat):
    # SMPL-X parents for joints 0..21 (pelvis + 21 body joints)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    local_mats = torch.cat([global_orient_mat[:, None], body_pose_mat], dim=1)  # (L, 22, 3, 3)
    global_mats = [local_mats[:, 0]]
    for j in range(1, len(parents)):
        global_mats.append(global_mats[parents[j]] @ local_mats[:, j])
    return torch.stack(global_mats, dim=1)


def merge_hamer_to_smplx(pred, mano_params):
    pred_out = copy.deepcopy(pred)
    smpl_incam = pred_out["smpl_params_incam"]
    smpl_global = pred_out["smpl_params_global"]

    # Left-hand convention conversion from HaMeR/MANO to SMPL-X.
    M = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float32))

    def _merge_one(smpl_params):
        body_pose = smpl_params["body_pose"].clone()  # (L, 63)
        global_orient = smpl_params["global_orient"].clone()  # (L, 3)
        L = body_pose.shape[0]
        dtype = body_pose.dtype
        device = body_pose.device

        if "left_hand_pose" in smpl_params:
            left_hand_pose = smpl_params["left_hand_pose"].clone()  # (L, 45)
        else:
            left_hand_pose = torch.zeros((L, 45), dtype=dtype, device=device)
        if "right_hand_pose" in smpl_params:
            right_hand_pose = smpl_params["right_hand_pose"].clone()  # (L, 45)
        else:
            right_hand_pose = torch.zeros((L, 45), dtype=dtype, device=device)

        if body_pose.shape[-1] < 63:
            raise ValueError(f"Expected body_pose with at least 63 dims, got shape {tuple(body_pose.shape)}")

        body_pose_mat = axis_angle_to_matrix(body_pose.reshape(-1, 3)).reshape(-1, 21, 3, 3)
        global_orient_mat = axis_angle_to_matrix(global_orient.reshape(-1, 3))
        global_mats = _compute_global_rotmats(global_orient_mat, body_pose_mat)
        left_elbow_global = global_mats[:, 18]
        right_elbow_global = global_mats[:, 19]

        left_valid = mano_params["left_hand_valid"].bool().to(device)
        right_valid = mano_params["right_hand_valid"].bool().to(device)

        left_go = mano_params["left_hand_global_orient"].float().to(device)
        right_go = mano_params["right_hand_global_orient"].float().to(device)
        left_hp = mano_params["left_hand_pose"].float().to(device)
        right_hp = mano_params["right_hand_pose"].float().to(device)

        M_local = M.to(device=device, dtype=dtype)
        left_go = M_local[None] @ left_go @ M_local[None]
        left_hp = M_local[None, None] @ left_hp @ M_local[None, None]

        # HaMeR global_orient is wrist global orientation in camera coordinates.
        left_wrist_global = left_go
        right_wrist_global = right_go
        left_wrist_local = torch.linalg.inv(left_elbow_global) @ left_wrist_global
        right_wrist_local = torch.linalg.inv(right_elbow_global) @ right_wrist_global

        left_wrist_aa = matrix_to_axis_angle(left_wrist_local)
        right_wrist_aa = matrix_to_axis_angle(right_wrist_local)
        left_hand_aa = matrix_to_axis_angle(left_hp.reshape(-1, 3, 3)).reshape(-1, 45)
        right_hand_aa = matrix_to_axis_angle(right_hp.reshape(-1, 3, 3)).reshape(-1, 45)

        body_pose[left_valid, 57:60] = left_wrist_aa[left_valid]
        body_pose[right_valid, 60:63] = right_wrist_aa[right_valid]
        left_hand_pose[left_valid] = left_hand_aa[left_valid]
        right_hand_pose[right_valid] = right_hand_aa[right_valid]

        smpl_params["body_pose"] = body_pose
        smpl_params["left_hand_pose"] = left_hand_pose
        smpl_params["right_hand_pose"] = right_hand_pose
        return smpl_params

    fused_incam = _merge_one(smpl_incam)
    pred_out["smpl_params_incam"] = fused_incam

    # Reuse fused local poses for global path to avoid mixing coordinate frames.
    if smpl_global is not None:
        if "body_pose" in smpl_global:
            smpl_global["body_pose"] = fused_incam["body_pose"].clone()
        smpl_global["left_hand_pose"] = fused_incam["left_hand_pose"].clone()
        smpl_global["right_hand_pose"] = fused_incam["right_hand_pose"].clone()
    pred_out["smpl_params_global"] = smpl_global
    pred_out["hamer_fused"] = True
    return pred_out


@torch.no_grad()
def run_preprocess(cfg):
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3) apply aspect ratio and enlarge
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)

    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # Get visual odometry results
    if not static_cam:  # use slam to get cam rotation
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
                vo_results = simple_vo.compute()  # (L, 4, 4), numpy
                torch.save(vo_results, paths.slam)
            else:  # DPVO
                from hmr4d.utils.preproc.slam import SLAMModel

                length, width, height = get_video_lwh(cfg.video_path)
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                bar = tqdm(total=length, desc="DPVO")
                while True:
                    ret = slam.track()
                    if ret:
                        bar.update()
                    else:
                        break
                slam_results = slam.process()  # (L, 7), numpy
                torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        if cfg.use_dpvo:  # DPVO
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        else:  # SimpleVO
            R_w2c = torch.from_numpy(traj[:, :3, :3])
    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }
    return data


def render_incam(cfg):
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results)
    # Use full 45D hand axis-angle so fused HaMeR hand poses can be rendered directly.
    smplx = make_smplx("supermotion", use_pca=False).cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    bbx_xys_render = torch.load(cfg.paths.bbx)["bbx_xys"]

    # -- render mesh -- #
    verts_incam = pred_c_verts
    writer = get_writer(incam_video_path, fps=30, crf=CRF)
    for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"):
        img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, [0.8, 0.8, 0.8])

        # # bbx
        # bbx_xys_ = bbx_xys_render[i].cpu().numpy()
        # lu_point = (bbx_xys_[:2] - bbx_xys_[2:] / 2).astype(int)
        # rd_point = (bbx_xys_[:2] + bbx_xys_[2:] / 2).astype(int)
        # img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)

        writer.write_frame(img)
    writer.close()
    reader.close()


def render_global(cfg):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results)
    # Use full 45D hand axis-angle so fused HaMeR hand poses can be rendered directly.
    smplx = make_smplx("supermotion", use_pca=False).cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    def move_to_start_point_face_z(verts):
        "XZ to origin, Start from the ground, Face-Z"
        # position
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        # face direction
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    # renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    render_length = length if not debug_cam else 8
    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in tqdm(range(render_length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()


if __name__ == "__main__":
    cfg, args = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    # ===== Preprocess and save to disk ===== #
    run_preprocess(cfg)
    run_hamer_preprocess(cfg, args)
    data = load_data_dict(cfg)

    # ===== HMR4D ===== #
    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        if args.use_hamer_hands:
            mano_params_path = Path(paths.get("mano_params", Path(cfg.preprocess_dir) / "mano_params.pt"))
            mano_params = torch.load(mano_params_path)
            pred = merge_hamer_to_smplx(pred, mano_params)
        data_time = data["length"] / 30
        Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
        torch.save(pred, paths.hmr4d_results)
    elif args.use_hamer_hands:
        pred = torch.load(paths.hmr4d_results)
        if not bool(pred.get("hamer_fused", False)):
            Log.info("[HMR4D] Existing results found without hand fusion. Applying cached HaMeR merge.")
            mano_params_path = Path(paths.get("mano_params", Path(cfg.preprocess_dir) / "mano_params.pt"))
            mano_params = torch.load(mano_params_path)
            pred = merge_hamer_to_smplx(pred, mano_params)
            torch.save(pred, paths.hmr4d_results)

    # ===== Render ===== #
    render_incam(cfg)
    render_global(cfg)
    if not Path(paths.incam_global_horiz_video).exists():
        Log.info("[Merge Videos]")
        merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)
