from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import os
import pandas as pd
import h5py
import yaml
from tqdm import tqdm
from types import SimpleNamespace
from collections import namedtuple

from utils.utils import*
from utils.eval_utils import initiate_model
from models.MLP import FiveClassClassifier, DiscriminativeAutoencoder
from models import ht_get_encoder
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from utils.file_utils import save_hdf5


parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="config_template.yaml")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def infer_single_slide(model, features, label, reverse_label_dict, model_type):
    """根据模型类型进行推理，并返回每个patch的预测类别（Y_hat）或相似度分数（Y_prob）。"""
    features = features.to(device)
    with torch.inference_mode():
        if model_type == 'MLP':
            logits = model(features)
        elif model_type == '自编码器':
            # 假设您只需要第二个返回值作为分类 logits
            _, logits, _ = model(features)
        else:
            # 默认使用 MLP 方式（或根据您的模型结构调整）
            logits = model(features)

        # 采用softmax转换为概率分布
        # 注意：这里直接返回概率分布 Y_prob 或其负值作为注意力分数可能更有用
        Y_prob = torch.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1).cpu().numpy()  # 获取每个patch的预测类别
        # 返回分类结果，或根据需要返回 Y_prob[:, 1].cpu().numpy() 作为热图分数

        return Y_prob



def load_params(df_entry, params):

    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]

            # 安全转换
            try:
                val = dtype(val)
            except:
                continue  # 跳过无法转换的值

            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
    return params


def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict



if __name__ == '__main__':

    config_path = os.path.join('heatmaps/configs', args.config_file)
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config_dict = parse_config_dict(args, config_dict)


    model_type = '聚类结果'  # 你的特定设置
    num_classes = 5
    input_dim, latent_dim = 1024, 64


    model = None
    if model_type == 'MLP':
        model = FiveClassClassifier(input_dim=1024, num_classes=num_classes)
        model.load_state_dict(torch.load('/home/idao/Zyf/models/oc_subtype_classficter/multi_view/samples_k3/fiveClassifier_60.pth'))
    elif model_type == '自编码器':
        model = DiscriminativeAutoencoder(input_dim, latent_dim, num_classes)
        model.load_state_dict(
            torch.load('/home/idao/Zyf/models/oc_subtype_classficter/Autoencoder/fold1/区分自编码器.pth'))

        model = DiscriminativeAutoencoder(input_dim, latent_dim, num_classes)
    elif model_type == '聚类结果':
        # 即使模型不用于推理，也需要加载它，因为它可能被 compute_from_patches 调用
        model = FiveClassClassifier(input_dim=1024, num_classes=num_classes)
        model.load_state_dict(
            torch.load('/home/idao/Zyf/models/oc_subtype_classficter/multi_view/samples_k3/fiveClassifier_100.pth'))
    model.to(device)
    model.eval()  # 确保模型处于评估模式

    # 打印配置
    print("--- 配置参数 ---")
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n' + key)
            for value_key, value_value in value.items():
                print(f"{value_key} : {value_value}")
        else:
            print(f"\n{key} : {value}")

    args_ns = SimpleNamespace(**config_dict)

    # 确保子字典也被转换为 SimpleNamespace 以方便访问
    patch_args = SimpleNamespace(**args_ns.patching_arguments)
    data_args = SimpleNamespace(**args_ns.data_arguments)
    model_args = SimpleNamespace(**args_ns.model_arguments)
    encoder_args = SimpleNamespace(**args_ns.encoder_arguments)
    exp_args = SimpleNamespace(**args_ns.exp_arguments)
    heatmap_args = SimpleNamespace(**args_ns.heatmap_arguments)
    sample_args = SimpleNamespace(**args_ns.sample_arguments)

    # 更新 model_args 中的 n_classes
    model_args.n_classes = exp_args.n_classes

    # 路径和尺寸定义

    PT_BASE_DIR = '/home/idao/Zyf/data/oc_features/FEATURES_DIRECTORY/pt_files'
    H5_BASE_DIR = '/home/idao/Zyf/data/oc_features/FEATURES_DIRECTORY/h5_files'
    SIM_BASE_DIR = 'heatmaps/sim'  # 注意力分数的基础路径

    patch_size = (patch_args.patch_size, patch_args.patch_size)
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print(
        f'patch_size: {patch_size[0]} x {patch_size[1]}, with {patch_args.overlap:.2f} overlap, step size is {step_size[0]} x {step_size[1]}')

    # 默认参数和数据帧初始化
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 'keep_ids': 'none',
                      'exclude_ids': 'none'}
    def_filter_params = {'a_t': 50.0, 'a_h': 8.0, 'max_n_holes': 10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    # 应用 preset
    if data_args.preset is not None:
        preset_df = pd.read_csv(data_args.preset)
        for params_dict in [def_seg_params, def_filter_params, def_vis_params, def_patch_params]:
            for key in params_dict.keys():
                params_dict[key] = preset_df.loc[0, key]

    # 初始化处理数据帧
    if data_args.process_list is None:

        data_dirs = data_args.data_dir if isinstance(data_args.data_dir, list) else [data_args.data_dir]
        slides = [slide for data_dir in data_dirs for slide in os.listdir(data_dir) if data_args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                           use_heatmap_args=False)
    else:
        df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                           use_heatmap_args=False)

    process_stack = df[df['process'] == 1].reset_index(drop=True)
    print(f'\nlist of slides to process (total: {len(process_stack)}): ')
    print(process_stack.head(len(process_stack)))

    # 特征提取器初始化 ---
    print('\nInitializing feature extractor')
    feature_extractor, img_transforms = ht_get_encoder(encoder_args.model_name,
                                                       target_img_size=encoder_args.target_img_size)
    feature_extractor.eval().to(device)
    print('Done!')

    # 标签字典 ---
    label_dict = data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))}

    # 目录创建 ---
    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)

    # WSI 循环处理 ---
    for i in tqdm(range(len(process_stack)), desc="Processing Slides"):
        row = process_stack.loc[i]
        slide_name = row['slide_id']
        if data_args.slide_ext not in slide_name:
            slide_name += data_args.slide_ext

        slide_id = slide_name.replace(data_args.slide_ext, '')
        print(f'\nprocessing: {slide_name}, ID: {slide_id}')

        label = row.get('label', 'Unspecified')
        grouping = reverse_label_dict.get(label, label)

        # 创建 WSI 专属保存目录
        p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping), slide_id)
        os.makedirs(p_slide_save_dir, exist_ok=True)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        # ROI 坐标
        if heatmap_args.use_roi:
            top_left = (int(row['x1']), int(row['y1']))
            bot_right = (int(row['x2']), int(row['y2']))
        else:
            top_left = None
            bot_right = None
        print(f'top left: {top_left}, bot right: {bot_right}')

        # slide_path
        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)
        elif isinstance(data_args.data_dir, dict):
            data_dir_key = row[data_args.data_dir_key]
            slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
        else:
            raise NotImplementedError

        mask_file = os.path.join(r_slide_save_dir, slide_id + '_mask.pkl')

        # 加载和处理参数
        seg_params = load_params(row, def_seg_params.copy())
        filter_params = load_params(row, def_filter_params.copy())
        vis_params = load_params(row, def_vis_params.copy())

        # 处理 keep_ids 和 exclude_ids
        for key in ['keep_ids', 'exclude_ids']:
            val = str(seg_params[key])
            if val and val != 'none':
                seg_params[key] = np.array(val.split(',')).astype(int)
            else:
                seg_params[key] = []

        # 打印参数
        print('Segmentation params:', seg_params)
        print('Filter params:', filter_params)
        print('Visualization params:', vis_params)

        # 初始化 WSI 对象
        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params,
                                    filter_params=filter_params)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
        vis_patch_size = tuple((np.array(patch_size) * wsi_ref_downsample * patch_args.custom_downsample).astype(int))

        # 检查和创建 WSI 掩膜图像
        block_map_save_path = os.path.join(r_slide_save_dir, f'{slide_id}_blockmap.h5')
        mask_path = os.path.join(r_slide_save_dir, f'{slide_id}_mask.jpg')
        if vis_params['vis_level'] < 0:
            vis_params['vis_level'] = wsi_object.wsi.get_best_level_for_downsample(32)

        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)
        wsi_object.saveSegmentation(mask_file)

        features_path = os.path.join(PT_BASE_DIR, slide_id + '.pt')
        h5_path = os.path.join(H5_BASE_DIR, slide_id + '.h5')

        if not os.path.isfile(features_path) and os.path.isfile(h5_path):
            with h5py.File(h5_path, "r") as file:
                features = torch.tensor(file['features'][:])
                torch.save(features, features_path)

        # 加载特征
        features = torch.load(features_path, weights_only=True)
        process_stack.loc[i, 'bag_size'] = len(features)


        if model_type == '聚类结果':
            sim_path = os.path.join(SIM_BASE_DIR, f"sim_{slide_id}.npy")
            if not os.path.isfile(sim_path):
                print(f"警告：未找到聚类结果文件 {sim_path}，跳过此 WSI。")
                del features
                continue

            Y_hats_all = np.load(sim_path)
            Y_hats = -Y_hats_all[:, 1]
            Y_hats = Y_hats.reshape(-1, 1)  # 确保维度正确
        else:
            # 使用模型进行推理
            Y_hats = infer_single_slide(model, features, label, reverse_label_dict, model_type)

        del features

        if not os.path.isfile(block_map_save_path):
            with h5py.File(h5_path, "r") as file:
                coords = file['coords'][:]

            # 使用 .cpu().numpy() 将 PyTorch 张量转换为 NumPy 数组
            if isinstance(Y_hats, torch.Tensor):
                asset_dict = {'attention_scores': Y_hats.cpu().numpy(), 'coords': coords}
            else:
                asset_dict = {'attention_scores': Y_hats, 'coords': coords}
            save_hdf5(block_map_save_path, asset_dict, mode='w')

        # 保存更新后的 process_stack
        csv_save_name = data_args.process_list.replace('.csv',
                                                       '') if data_args.process_list is not None else exp_args.save_exp_code
        process_stack.to_csv(f'heatmaps/results/{csv_save_name}.csv', index=False)

        # 读取 block map
        with h5py.File(block_map_save_path, 'r') as file:
            scores = file['attention_scores'][:]
            coords = file['coords'][:]

        wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size,
                      'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                      'use_center_shift': heatmap_args.use_center_shift}

        heatmap_raw_save_path = os.path.join(r_slide_save_dir, f'{slide_id}_blockmap.png')
        if not os.path.isfile(heatmap_raw_save_path):

            if model_type=="MLP":
                heatmap = drawHeatmap(model_type, scores, coords, slide_path, wsi_object=wsi_object,
                                  alpha=heatmap_args.alpha, use_holes=True,
                                vis_level=-1, blank_canvas=False,
                                  patch_size=vis_patch_size)
            else:
                heatmap = drawHeatmap(model_type, scores, coords, slide_path, wsi_object=wsi_object,
                                      cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True,
                                      binarize=False, vis_level=-1, blank_canvas=False, thresh=-1,
                                      patch_size=vis_patch_size, convert_to_percentiles=True)
            heatmap.save(heatmap_raw_save_path)
            del heatmap

        save_path = os.path.join(r_slide_save_dir, f'{slide_id}_{patch_args.overlap}_roi_{heatmap_args.use_roi}.h5')

        ref_scores = scores if heatmap_args.use_ref_scores else None

        # 重新计算热图（如果需要）
        if heatmap_args.calc_heatmap:
            compute_from_patches(wsi_object=wsi_object, img_transforms=img_transforms,
                                 clam_pred=Y_hats[0], model=model,
                                 feature_extractor=feature_extractor,
                                 batch_size=exp_args.batch_size, **wsi_kwargs,
                                 attn_save_path=save_path, ref_scores=ref_scores)

        # 检查并加载精细热图分数
        if not os.path.isfile(save_path):
            print(f'heatmap {save_path} not found.')

            if heatmap_args.use_roi:
                save_path_full = os.path.join(r_slide_save_dir, f'{slide_id}_{patch_args.overlap}_roi_False.h5')
                if os.path.isfile(save_path_full):
                    print('Found heatmap for whole slide as fallback.')
                    save_path = save_path_full
                else:
                    continue
            else:
                continue

        with h5py.File(save_path, 'r') as file:
            scores = file['attention_scores'][:]
            coords = file['coords'][:]

        # 绘制最终精细热图
        heatmap_vis_args = {'convert_to_percentiles': not heatmap_args.use_ref_scores,
                            'vis_level': heatmap_args.vis_level,
                            'blur': heatmap_args.blur,
                            'custom_downsample': heatmap_args.custom_downsample}

        heatmap_save_name = '{}_{:.1f}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{:.1f}.{}'.format(
            slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
            int(heatmap_args.blur), int(heatmap_args.use_ref_scores),
            int(heatmap_args.blank_canvas), float(heatmap_args.alpha),
            int(heatmap_args.vis_level), int(heatmap_args.binarize),
            float(heatmap_args.binary_thresh), heatmap_args.save_ext)

        heatmap_final_save_path = os.path.join(p_slide_save_dir, heatmap_save_name)

        if not os.path.isfile(heatmap_final_save_path):

            heatmap = drawHeatmap(model_type, scores, coords, slide_path, wsi_object=wsi_object,
                                  cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args,
                                  binarize=heatmap_args.binarize,
                                  blank_canvas=heatmap_args.blank_canvas,
                                  thresh=heatmap_args.binary_thresh, patch_size=vis_patch_size,
                                  overlap=patch_args.overlap, top_left=top_left, bot_right=bot_right)

            save_kwargs = {'quality': 100} if heatmap_args.save_ext == 'jpg' else {}
            heatmap.save(heatmap_final_save_path, **save_kwargs)

        # 保存原始 WSI 图像 ---
        if heatmap_args.save_orig:
            vis_level = heatmap_args.vis_level if heatmap_args.vis_level >= 0 else vis_params['vis_level']
            heatmap_save_name = f'{slide_id}_orig_{int(vis_level)}.{heatmap_args.save_ext}'
            heatmap_orig_save_path = os.path.join(p_slide_save_dir, heatmap_save_name)

            if not os.path.isfile(heatmap_orig_save_path):
                heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True,
                                            custom_downsample=heatmap_args.custom_downsample)
                save_kwargs = {'quality': 100} if heatmap_args.save_ext == 'jpg' else {}
                heatmap.save(heatmap_orig_save_path, **save_kwargs)

    # 结束：保存配置 ---
    with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)