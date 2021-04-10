import os
import fdj
import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from himawari import Himawari
from database import Database


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
def fire_detection(pk_path, args_dict, himawari):
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
    num_of_cloud_edge = args_dict['num_of_cloud_edge']
    band7_minimum_th = args_dict['band7_minimum_th']

    # 夜间判识条件
    band7_std_night_th = args_dict['band7_std_night_th']
    band7_14_night_th = args_dict['band7_14_night_th']

    # 从pk文件中，读取经纬度，和16个波段的影像
    h8_band2 = himawari.ch2_refl
    h8_band3 = himawari.ch3_refl
    h8_band4 = himawari.ch4_refl
    h8_band6 = himawari.ch6_refl
    h8_band7 = himawari.ch7_brt
    h8_band14 = himawari.ch14_brt
    h8_band15 = himawari.ch15_brt
    h8_band7_14 = himawari.ch7_14_brt
    h8_band3_plus_4 = h8_band3 + h8_band4
    lon_list = himawari.longitude_arr
    lat_list = himawari.latitude_arr
    pk_filename = himawari.filename
    curr_datetime = datetime.datetime.strptime(pk_filename[4:], "%Y%m%d%H%M")

    # 读取上一帧的pk影像，用于绝对火点判定
    band7_deta = np.array([])
    band7_14_deta = np.array([])
    last_datetime = curr_datetime - datetime.timedelta(seconds=0, minutes=10, hours=0)
    last_datetime = last_datetime.strftime('%Y%m%d%H%M')
    pk_last_path = os.path.join(pk_path, "SHJC" + last_datetime + ".pk")
    if os.path.exists(pk_last_path):
        himawari_last = Himawari(pk_last_path)
        band7_deta = himawari.ch7_brt - himawari_last.ch7_brt
        band7_14_deta = himawari.ch7_14_brt - himawari_last.ch7_14_brt

    # 将搜索区域缩小至云南范围内
    yunnan_min_lon_idx = int((97.50 - 96.0) / 0.01)
    yunnan_max_lon_idx = int((106.20 - 96.0) / 0.01)
    yunnan_min_lat_idx = int((30 - 29.26) / 0.01)
    yunnan_max_lat_idx = int((30 - 21.12) / 0.01)
    pixel_num = ((106.20 - 97.50) / 0.01 + 1) * ((29.26 - 21.12) / 0.01 + 1)

    # 云掩膜
    night_mask = np.logical_and(h8_band3 < 0.01, h8_band4 < 0.01)
    condition1 = np.logical_and((h8_band3 + h8_band4) < 0.36, h8_band15 > 265)
    condition2 = np.logical_or((h8_band3 + h8_band4) < 0.32, h8_band15 > 285)
    cloud_mask = np.logical_not(np.logical_or(np.logical_and(condition1, condition2), night_mask))
    cloud_img = np.array(cloud_mask, dtype=int)

    num = 0
    print(f'Processing:{pk_filename}')
    for lat_idx in range(yunnan_min_lat_idx, yunnan_max_lat_idx + 1):
        for lon_idx in range(yunnan_min_lon_idx, yunnan_max_lon_idx + 1):
            num = num + 1
            if num % 50000 == 0:
                print("\rProgress: " + str(round(num * 100 / pixel_num, 2)) + "%", end="")

            # 若该像元为云像元，跳过
            if cloud_img[lat_idx, lon_idx] == 1:
                continue
            # 该像元背景像元的云像元比例超过25%，直接判断为非火点(有效像元不足75%)
            size = 15
            kernal_point = int((size - 1) / 2)
            x_start = lat_idx - kernal_point
            y_start = lon_idx - kernal_point
            kernel_cloud = cloud_img[x_start:x_start + size + 1, y_start:y_start + size + 1]
            if np.sum(kernel_cloud) / (15 * 15) >= 1 - valid_pxl_ratio:
                continue
            # 云边检测
            kernal_point = int((5 - 1) / 2)
            x_start = lat_idx - kernal_point
            y_start = lon_idx - kernal_point
            kernel_cloud_edge = cloud_img[x_start:x_start + 5, y_start:y_start + 5]
            if np.sum(kernel_cloud_edge) > num_of_cloud_edge:
                continue
            # 排除7通道亮温值过低的火点
            if h8_band7[lat_idx, lon_idx] <= band7_minimum_th:
                continue
            # 计算像元的若干特征，用于火点判别
            # n3 - 通道3反射率; n4 - 通道7亮温值; n5 - 通道7减通道14亮温差值;
            # n1_std_with_valid_pixels - 通道7亮温值标准化结果; n2_std_with_valid_pixels - 通道7减通道14亮温差值标准化结果;
            # band7_14_mean - 通道7减通道14亮温均值; nightBool - 夜判结果; cloudBool - 云判结果; waterBool - 水判结果;
            # band7_mean_size7 - 小窗口通道7均值; band7_14_mean_size7 - 小窗口通道7减通道14亮温均值;
            # n_cloud - 通道3减通道4反射率标准化结果, n_cloud_band2 - 通道2反射率标准化结果
            n3, n4, n5, n1_std_with_valid_pixels, n2_std_with_valid_pixels, band7_14_mean, nightBool, cloudBool, waterBool,\
            band7_mean_size7, band7_14_mean_size7, n_cloud, n_cloud_band2 = fdj.hd_test(h8_band3,
                                                                                        h8_band7,
                                                                                        h8_band14,
                                                                                        lat_idx,
                                                                                        lon_idx,
                                                                                        h8_band4,
                                                                                        h8_band15,
                                                                                        h8_band6,
                                                                                        h8_band3_plus_4,
                                                                                        h8_band2, kernel_cloud,
                                                                                        size=15
                                                                                        )
            band3_data = h8_band3[lat_idx, lon_idx]  # 获取3通道反射率
            band4_data = h8_band4[lat_idx, lon_idx]  # 获取4通道反射率
            band3plus4_data = band3_data + band4_data  # 计算通道3与通道4反射率的和，用于判云
            datetime_tmp = curr_datetime.strftime("%Y/%m/%d %H:%M")
            # 绝对火点判据
            if n4 > band7_absolute_th:
                fire_absolute_df = fire_absolute_df.append({
                    'Time': datetime_tmp, 'Lons': lon_list[lon_idx],
                    'Lats': lat_list[lat_idx], 'band7': n4,
                    'band7_deta': band7_deta, 'band7-14_deta': band7_14_deta
                }, ignore_index=True)
            if band7_deta.size > 0 and band7_14_deta.size > 0 and band7_deta[lat_idx][lon_idx] > band7_incre_th and band7_14_deta[lat_idx][lon_idx] > band7_14_incre_th:
                fire_absolute_df = fire_absolute_df.append({
                    'Time': datetime_tmp, 'Lons': lon_list[lon_idx],
                    'Lats': lat_list[lat_idx], 'band7': n4,
                    'band7_deta': band7_deta, 'band7-14_deta': band7_14_deta
                }, ignore_index=True)

            # 基于背景像元的相对火点判据
            n1plusn2 = n1_std_with_valid_pixels + n2_std_with_valid_pixels
            condition1 = 6.34 <= n1plusn2 < 7 and (n1plusn2 + band7_14_mean_size7 > 14) and n1_std_with_valid_pixels > 2.91 and n5 > 12.3
            condition2 = band7_14_mean_size7 > 6.57 and 6.35 < n1plusn2 < 7.0 and n1_std_with_valid_pixels > 2.91 and n5 > 12.3
            condition3 = 7.0 <= n1plusn2 < 8 and n1plusn2 + band7_14_mean_size7 > 13.4 and n1_std_with_valid_pixels > 2.42 and n5 > 10.2
            condition4 = band7_14_mean_size7 > 4.4 and 7.61 < n1plusn2 < 8.0 and n1_std_with_valid_pixels > 3.47 and n5 > 8.4
            condition5 = 8.0 <= n1plusn2 < 9 and n1plusn2 + band7_14_mean_size7 > 12.38 and n1_std_with_valid_pixels > 2.42 and n5 > 8.8
            condition6 = band7_14_mean_size7 > 2.1 and 8 < n1plusn2 < 9.0 and n2_std_with_valid_pixels > 4.26 and n5 > 6.6
            condition7 = 9.0 <= n1plusn2 < 10.0 and n1plusn2 + band7_14_mean_size7 > 10.4 and n1_std_with_valid_pixels > 4 and n5 > 7.1
            condition8 = n1plusn2 >= 10.0 and n1plusn2 + band7_14_mean_size7 > 0.8 and n1_std_with_valid_pixels > 4.35 and n5 > 6.0
            if condition1 or condition2 or condition3 or condition4 or condition5 or condition6 or condition7 or condition8:
                fire_relative_df = fire_relative_df.append(
                    {'Time': datetime_tmp, 'Lons': lon_list[lon_idx], 'Lats': lat_list[lat_idx],
                     'cloud?': cloudBool, 'water?': waterBool, 'night?': nightBool,
                     'n1_std_valid': n1_std_with_valid_pixels, 'n2_std_valid': n2_std_with_valid_pixels, 'n3': n3,
                     'n4': n4, 'n5': n5, 'n1+n2': n1plusn2,
                     'band7_deta': band7_deta, 'band7-14_deta': band7_14_deta, 'band7_mean_size7': band7_mean_size7,
                     'band7-14_mean_size7': band7_14_mean_size7, 'band3+4': band3plus4_data,
                     'n_cloud': n_cloud, 'n_cloud_band2': n_cloud_band2},
                    ignore_index=True)
                continue
            if nightBool and n2_std_with_valid_pixels > band7_std_night_th and n3 < band3_th and band3plus4_data < band3_plus_4_th and n4 > band7_relative_night_th and n5 > band7_14_night_th:
                fire_relative_df = fire_relative_df.append(
                    {'Time': datetime_tmp, 'Lons': lon_list[lon_idx], 'Lats': lat_list[lat_idx],
                     'cloud?': cloudBool, 'water?': waterBool, 'night?': nightBool,
                     'n1_std_valid': n1_std_with_valid_pixels, 'n2_std_valid': n2_std_with_valid_pixels, 'n3': n3,
                     'n4': n4, 'n5': n5, 'n1+n2': n1plusn2,
                     'band7_deta': band7_deta, 'band7-14_deta': band7_14_deta, 'band7_mean_size7': band7_mean_size7,
                     'band7-14_mean_size7': band7_14_mean_size7, 'band3+4': band3plus4_data,
                     'n_cloud': n_cloud, 'n_cloud_band2': n_cloud_band2},
                    ignore_index=True)
                continue

    if fire_relative_df.shape[0] <= 1:
        return
    # 对火点进行聚类操作
    fire_lons = fire_relative_df['Lons'].values.reshape(-1)
    fire_lats = fire_relative_df['Lats'].values.reshape(-1)
    if len(fire_lons) >= 2 or len(fire_lats) >= 2:
        comb_lons = list()
        comb_lats = list()

        resize_ptrs = np.array([[item_x, item_y] for item_x, item_y in zip(fire_lons, fire_lats)])
        ptrs_clusters = DBSCAN(eps=0.03, min_samples=2).fit(resize_ptrs)
        max_cluster_label = np.max(ptrs_clusters.labels_)
        for cluster_i in np.arange(0, max_cluster_label + 1):
            cur_clust_elmidxs = np.where(ptrs_clusters.labels_ == cluster_i)[0]     # 获取当前标签为i的聚类元素索引值
            cur_clust_avglon = np.mean(resize_ptrs[cur_clust_elmidxs, 0])           # 计算当前聚类平均经度
            cur_clust_avglat = np.mean(resize_ptrs[cur_clust_elmidxs, 1])           # 计算当前聚类平均纬度
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

    # 保存绝对火点
    if fire_absolute_df.shape[0] > 0:
        fire_absolute_csv = 'SHJC' + curr_datetime.strftime('%Y%m%d%H%M') + '5_absolute.csv'
        fire_absolute_df.to_csv("./fire_detection/output_absolute/" + fire_absolute_csv)

    # 根据火点聚类，筛出面积过大的火点
    if fire_relative_df.shape[0] > 0:
        fire_relative_df.to_csv(f"./fire_detection/output/{pk_filename}5.csv", index=False)
        fire_eliminate_df = cloudEliminate(fire_relative_df, num_of_cluster_th)
        if fire_eliminate_df.shape[0] > 0:
            fire_eliminate_df.to_csv(f"./fire_detection/output_eliminate/{pk_filename}5.csv", index=False)


# pk_src_path - pk文件路径; img_dst_path - 图片存储路径; params - 算法参数
def main(pk_src_path, img_dst_path, params):

    # 山火识别特征参数阈值
    args_dict = {
        "band3_th": 0.13,  # 3通道阈值
        "band3_plus_4_th": 0.3415,  # 3+4通道阈值
        "band7_relative_night_th": 283,
        "band7_std_night_th": 4.39,  # 夜间判别
        "band7_14_night_th": 3.3,  # 7通道-14通道 阈值，中红外-远红外
        "num_of_cluster_th": 20,  # 一片火的像素个数不超过cluster_limit
        # 可调参数
        "valid_pxl_ratio": params["有效像元比例"],
        "num_of_cloud_edge": params["云边判识云像元数量"],
        "band7_minimum_th": params["非火点阈值"],
        "band7_absolute_th": params["绝对火点阈值"],
        "band7_incre_th": params["7通道亮温增量"],
        "band7_14_incre_th": params["7-14通道亮温增量"]
    }

    # 读取记录数据，防止重复处理pk文件
    if not os.path.exists("./record.txt"):
        record_arr = np.array([])
    else:
        record_arr = np.loadtxt("./record.txt", delimiter=",", dtype=str)
    pk_all_list = [file for file in os.listdir(pk_src_path) if file.endswith(".pk")]

    # 对未处理的pk文件进行火点监测
    file_list = np.sort(np.setdiff1d(pk_all_list, record_arr))
    for filename in file_list:
        pk_path = os.path.join(pk_src_path, filename)
        h = Himawari(pk_path)

        # 保存himawari-8卫星图像至数据库
        database = Database("121.37.246.140", "FireDetection", "gdut", "fire#319")
        output_path_list = h.create_img(img_dst_path)
        database.img_insert(output_path_list, h.filename)
        database.close()

        fire_detection(pk_src_path, args_dict, h)     # 火点判识程序

    np.savetxt("./record.txt", pk_all_list, fmt="%s", delimiter=",")
    return file_list


if __name__ == '__main__':
    fire_detection()
