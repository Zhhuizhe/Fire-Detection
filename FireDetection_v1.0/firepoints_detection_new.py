import os
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from himawari import Himawari


def mat2col(mat, kernel_size):
    mat = np.pad(mat, int((kernel_size - 1) / 2), constant_values=0)
    rows, cols = mat.shape
    col_extent = cols - kernel_size + 1
    row_extent = rows - kernel_size + 1

    start_idx = np.arange(kernel_size)[:, None] * cols + np.arange(kernel_size)
    offset_idx = np.arange(row_extent)[:, None] * cols + np.arange(col_extent)

    return np.take(mat, (start_idx.ravel()[:, None] + offset_idx.ravel()[:]))


def cloudEliminate(fire_relative_df, threshold):
    fire_lons = fire_relative_df['Lons'].values
    fire_lats = fire_relative_df['Lats'].values
    if len(fire_lons) >= 2 or len(fire_lats) >= 2:
        comb_lons = list()
        comb_lats = list()

        resize_ptrs = np.vstack((fire_lons, fire_lats)).T
        ptrs_clusters = DBSCAN(eps=0.03, min_samples=2).fit(resize_ptrs)  # dbscan聚类算法

        max_cluster_label = np.max(ptrs_clusters.labels_)
        for cluster_i in np.arange(0, max_cluster_label + 1):
            cur_clust_elmidxs = np.where(ptrs_clusters.labels_ == cluster_i)[0]
            cur_clust_avglon = np.mean(resize_ptrs[cur_clust_elmidxs, 0])
            cur_clust_avglat = np.mean(resize_ptrs[cur_clust_elmidxs, 1])

            comb_lons.append(round(cur_clust_avglon, 2))
            comb_lats.append(round(cur_clust_avglat, 2))

        fire_lons_dbscan = comb_lons
        fire_lats_dbscan = comb_lats
        output_dbscan = pd.DataFrame()
        output_dbscan['Lons'] = fire_lons_dbscan
        output_dbscan['Lats'] = fire_lats_dbscan
        output_dbscan['Time'] = fire_relative_df['Time'][0]
        time_str = fire_relative_df['Time'][0]
        time_str = datetime.datetime.strftime(datetime.datetime.strptime(time_str, "%Y/%m/%d %H:%M"), '%Y%m%d%H%M')
        output_dbscan.to_csv('./fire_detection/output_dbscan/' + 'SHJC' + time_str + '_dbscan.csv')
        fire_relative_df['dbscan'] = ptrs_clusters.labels_
    else:
        fire_relative_df['dbscan'] = -1
    lables = np.unique(fire_relative_df['dbscan'].values).tolist()
    output_temp_new = pd.DataFrame()
    for lable in lables:
        block = fire_relative_df[fire_relative_df['dbscan'] == lable]         # 选取output_temp中所有标签为label的数据
        if block['n_cloud'].max() < 9 and block['n_cloud_band2'].max() < 10:
            if block.shape[0] < threshold:
                output_temp_new = pd.concat([output_temp_new, block], axis=0)
    return output_temp_new


# 对POI_time时间的影像pk文件，做火点检测
def fire_detection(pk_path):
    start_time = time.time()

    h = Himawari("./fire_detection/H8_pk/SHJC202102231630.pk")

    args_dict = {
        "band3_th": 0.13,  # 3通道阈值
        "band3_plus_4_th": 0.3415,  # 3+4通道阈值
        "band7_relative_night_th": 283,
        "band7_std_night_th": 4.39,  # 夜间判别
        "band7_14_night_th": 3.3,  # 7通道-14通道 阈值，中红外-远红外
        "num_of_cluster_th": 20,
        "window_size": 15,
        "cloud_edge_window_size": 5,
        # 可调参数
        "valid_pxl_ratio": 0.7,
        "num_of_cloud_edge": 6,
        "band7_minimum_th": 260,
        "band7_absolute_th": 330,
        "band7_incre_th": 20,
        "band7_14_incre_th": 20
    }

    fire_absolute_df = pd.DataFrame()       # 存储绝对火点
    fire_relative_df = pd.DataFrame()       # 存储相对火点像元

    # 从argsDict中获取多组阈值
    band3_th = args_dict['band3_th']
    band7_relative_night_th = args_dict['band7_relative_night_th']
    band3_plus_4_th = args_dict['band3_plus_4_th']
    band7_absolute_th = args_dict['band7_absolute_th']
    band7_incre_th = args_dict['band7_incre_th']
    band7_14_incre_th = args_dict['band7_14_incre_th']
    num_of_cluster_th = args_dict['num_of_cluster_th']
    valid_pxl_ratio = args_dict['valid_pxl_ratio']
    band7_minimum_th = args_dict['band7_minimum_th']

    # 夜间判识条件
    band7_std_night_th = args_dict['band7_std_night_th']
    band7_14_night_th = args_dict['band7_14_night_th']

    # 从pk文件中，读取经纬度，和16个波段的影像
    h8_band2 = h.ch2_refl
    h8_band3 = h.ch3_refl
    h8_band4 = h.ch4_refl
    h8_band6 = h.ch6_refl
    h8_band7 = h.ch7_brt
    h8_band14 = h.ch14_brt
    h8_band15 = h.ch15_brt
    h8_band7_14 = h.ch7_14_brt
    h8_band3_plus_4 = h8_band3 + h8_band4
    lon_list = h.longitude_arr
    lat_list = h.latitude_arr
    pk_filename = h.filename
    curr_datetime = datetime.datetime.strptime(pk_filename[4:], "%Y%m%d%H%M")

    # 读取上一帧的pk影像，用于绝对火点判定
    band7_deta = np.array([])
    band7_14_deta = np.array([])
    last_datetime = curr_datetime - datetime.timedelta(seconds=0, minutes=10, hours=0)
    last_datetime = last_datetime.strftime('%Y%m%d%H%M')
    pk_last_path = os.path.join(pk_path, "SHJC" + last_datetime + ".pk")
    if os.path.exists(pk_last_path):
        himawari_last = Himawari(pk_last_path)
        band7_deta = h.ch7_brt - himawari_last.ch7_brt
        band7_14_deta = h.ch7_14_brt - himawari_last.ch7_14_brt

    # 云掩膜
    night_mask = np.logical_and(h8_band3 < 0.01, h8_band4 < 0.01)
    condition1 = np.logical_and((h8_band3 + h8_band4) < 0.36, h8_band15 > 265)
    condition2 = np.logical_or((h8_band3 + h8_band4) < 0.32, h8_band15 > 285)
    cloud_mask = np.logical_not(np.logical_or(np.logical_and(condition1, condition2), night_mask))
    cloud_img = np.array(cloud_mask, dtype=int)

    # 有效像元判识
    ws = args_dict["window_size"]
    if not ws or ws % 2 == 0:
        print(f"Wrong window size, window size must be a odd number(window size:{ws})")
        return
    mat_tmp = mat2col(cloud_img, args_dict["window_size"])
    n_cloud_pxl_mat = np.sum(mat_tmp, axis=0).reshape((1000, 1100))
    invalid_pxl_mat = (n_cloud_pxl_mat / (15 * 15) > 1 - valid_pxl_ratio)

    # 云边判识
    cloud_edge_ws = args_dict["window_size"]
    if not cloud_edge_ws or cloud_edge_ws % 2 == 0:
        print(f"Wrong window size, window size must be a odd number(window size:{cloud_edge_ws})")
        return
    offset = (ws - cloud_edge_ws) / 2
    idx_arr = np.arange(offset * (ws + 1), offset * (ws + 1) + cloud_edge_ws - 1, 1)
    n_cloud_pxl_mat = np.sum(mat_tmp, axis=0)
    cloud_edge_pxl_mat = n_cloud_pxl_mat > args_dict["num_of_cloud_edge"]

    # 排除低温像元
    non_fire_mat = h8_band7 < band7_minimum_th

    # 将搜索区域缩小至云南范围内
    yunnan_min_lon_idx = int((97.50 - 96.0) / 0.01)
    yunnan_max_lon_idx = int((106.20 - 96.0) / 0.01)
    yunnan_min_lat_idx = int((30 - 29.26) / 0.01)
    yunnan_max_lat_idx = int((30 - 21.12) / 0.01)

    th_mat = cloud_img + invalid_pxl_mat + invalid_pxl_mat + cloud_edge_pxl_mat + non_fire_mat
    valid_pxl_pos = np.where(th_mat == 0)


    # valid_pxl_pos = np.vstack((valid_pxl_pos[0], valid_pxl_pos[1])).T

    # 绝对火点判断
    fire_absolute_arr = h8_band7[valid_pxl_pos[0], valid_pxl_pos[1]] > band7_absolute_th
    if band7_deta.size > 0 and band7_14_deta.size > 0:
        fire_absolute_arr = \
            np.logical_or(fire_absolute_arr, np.logical_and(band7_deta > band7_incre_th, band7_14_deta > band7_14_incre_th))

    # 相对火点判断，ws(window size)窗口大小
    # h8_band7_14 = h8_band7_14[valid_pxl_pos[0], valid_pxl_pos[1]]
    band7_ws15 = mat2col(h8_band7, 15)
    band7_14_ws15 = mat2col(h8_band7_14, 15)
    band7_14_ws7 = mat2col(h8_band7_14, 7)
    band7_ws15_norm = ((h8_band7 - np.mean(band7_ws15, axis=0).reshape((1000, 1100))) / np.std(band7_ws15, axis=0).reshape((1000, 1100)))
    band7_14_ws15_norm = ((h8_band7_14 - np.mean(band7_14_ws15, axis=0).reshape((1000, 1100))) / np.std(band7_14_ws15, axis=0).reshape((1000, 1100)))
    band7_14_ws7_mean = np.mean(band7_14_ws7, axis=0).reshape((1000, 1100))

    ws15_norm_sum = band7_ws15_norm + band7_14_ws15_norm
    th_dict = {
        "th1": {"con1": (6.34, 7), "con2": 14, "con3": 2.91, "con4": 12.3},
        "th2": {"con1": (7, 8), "con2": 13.4, "con3": 2.42, "con4": 10.2},
        "th3": {"con1": (8, 9), "con2": 12.38, "con3": 2.42, "con4": 8.8},
        "th4": {"con1": (9, 10), "con2": 10.4, "con3": 4, "con4": 7.1},
        "th5": {"con1": (10, 10000), "con2": 0.8, "con3": 4.35, "con4": 6.0}
    }
    mat_tmp = ws15_norm_sum + band7_14_ws7_mean
    # if (6.34 <= ws15_norm_sum < 7 and (ws15_norm_sum + band7_14_ws7_mean > 14) and band7_ws15_norm > 2.91 and h8_band7_14 > 12.3) or \
    #     (7.0 <= ws15_norm_sum < 8 and ws15_norm_sum + band7_14_ws7_mean > 13.4 and band7_ws15_norm > 2.42 and h8_band7_14 > 10.2) or \
    #     (8.0 <= ws15_norm_sum < 9 and ws15_norm_sum + band7_14_ws7_mean > 12.38 and band7_ws15_norm > 2.42 and h8_band7_14 > 8.8) or \
    #     (9.0 <= ws15_norm_sum < 10.0 and ws15_norm_sum + band7_14_ws7_mean > 10.4 and band7_ws15_norm > 4 and h8_band7_14 > 7.1) or \
    #     (ws15_norm_sum >= 10.0 and ws15_norm_sum + band7_14_ws7_mean > 0.8 and band7_ws15_norm > 4.35 and h8_band7_14 > 6.0):
    #     pass
    ret = np.zeros((1000, 1100))
    for val in dict_tmp.values():
        con1 = np.logical_and(val["con1"][0] <= ws15_norm_sum, ws15_norm_sum < val["con1"][1])
        con2 = (ws15_norm_sum + band7_14_ws7_mean > val["con2"])
        con3 = (band7_ws15_norm > val["con3"])
        con4 = (h8_band7_14 > val["con4"])
        print(len(np.where(con1 + con2 + con3 + con4 - 4 > 0)[0]))
        print(con1 + con2 + con3 + con4 - 4)
        ret = np.logical_or(ret, (con1 + con2 + con3 + con4 - 4) > 0)

    con1 = ((band7_14_ws7_mean > 6.57) + np.logical_and(6.35 < ws15_norm_sum, ws15_norm_sum < 7.0) + (band7_ws15_norm > 2.91) + (h8_band7_14 > 12.3) - 4) > 0
    con2 = ((band7_14_ws7_mean > 4.4) + np.logical_and(7.61 < ws15_norm_sum, ws15_norm_sum < 8.0) + (band7_ws15_norm > 3.47) + (h8_band7_14 > 8.4) - 4) > 0
    con3 = ((band7_14_ws7_mean > 2.1) + np.logical_and(8 < ws15_norm_sum, ws15_norm_sum < 9.0) + (band7_14_ws15_norm > 4.26) + (h8_band7_14 > 6.6) - 4) > 0
    ret = np.logical_or(ret, np.logical_or(con1, (np.logical_or(con3, con2))))

    ret = np.logical_and(ret, np.logical_not(th_mat))


    # if nightBool and n2_std_with_valid_pixels > band7_std_night_th and h8_band3 < band3_th and h8_band3_plus_4 < band3_plus_4_th and h8_band7 > band7_relative_night_th and h8_band7_14 > band7_14_night_th:
    #     pass
    datetime_tmp = curr_datetime.strftime("%Y/%m/%d %H:%M")
    (lat_idx_arr, lon_idx_arr) = np.where(ret == 1)
    total = len(lat_idx_arr)
    print(total)
    num = 0
    for lat_idx, lon_idx in zip(lat_idx_arr, lon_idx_arr):
        fire_relative_df = fire_relative_df.append(
            {'Time': datetime_tmp, 'Lons': lon_list[lon_idx], 'Lats': lat_list[lat_idx]},
            ignore_index=True)
        num += 1
        print("\rProgress: " + str(round(num * 100 / total, 2)) + "%", end="")
    end_time = time.time()
    print((end_time - start_time)/60)


if __name__ == '__main__':
    # fire_detection("./Temp/SHJC202103131530.pk")
    a = [1, 2, 3]
    print(a[:, None] + [1, 2, 3])
