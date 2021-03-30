import os
import sys
import pickle
import fdj
import datetime
import numpy as np
import pandas as pd
import cv2 as cv
from sklearn.cluster import DBSCAN
from scipy import misc

if './' not in sys.path:
    sys.path.insert(0, './')


def imgProcess(folder_path, fire_index, truth_lon, truth_lat):  # 被imgGeneration调用，把火点像元用 圆圈 ”○“ 标出来
    img_band2 = misc.imread(folder_path + '/band2' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg')
    img_band3 = misc.imread(folder_path + '/band3' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg')
    img_band4 = misc.imread(folder_path + '/band4' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg')
    img_band7 = misc.imread(folder_path + '/band7' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg')
    img_band7_14 = misc.imread(folder_path + '/band7_14' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg')

    img_band2 = cv.resize(img_band2, None, fx=50, fy=50, interpolation=cv.INTER_NEAREST)
    img_band2 = cv.circle(img_band2, (fire_index[0] * 50, fire_index[1] * 50), 10, (0, 0, 255))
    misc.imsave(folder_path + '/band2' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg', img_band2)

    img_band3 = cv.resize(img_band3, None, fx=50, fy=50, interpolation=cv.INTER_NEAREST)
    img_band3 = cv.circle(img_band3, (fire_index[0] * 50, fire_index[1] * 50), 10, (0, 0, 255))
    misc.imsave(folder_path + '/band3' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg', img_band3)

    img_band4 = cv.resize(img_band4, None, fx=50, fy=50, interpolation=cv.INTER_NEAREST)
    img_band4 = cv.circle(img_band4, (fire_index[0] * 50, fire_index[1] * 50), 10, (0, 0, 255))
    misc.imsave(folder_path + '/band4' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg', img_band4)

    img_band7 = cv.resize(img_band7, None, fx=50, fy=50, interpolation=cv.INTER_NEAREST)
    img_band7 = cv.circle(img_band7, (fire_index[0] * 50, fire_index[1] * 50), 10, (0, 0, 255))
    misc.imsave(folder_path + '/band7' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg', img_band7)

    img_band7_14 = cv.resize(img_band7_14, None, fx=50, fy=50, interpolation=cv.INTER_NEAREST)
    img_band7_14 = cv.circle(img_band7_14, (fire_index[0] * 50, fire_index[1] * 50), 10, (0, 0, 255))
    misc.imsave(folder_path + '/band7_14' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg', img_band7_14)


def imgGeneration(folder_path, lon, lat, lon_list, lat_list, truth_time, h8_band2_data, h8_band3_data, h8_band4_data,
                  h8_band7_data, h8_band7_14_data, img_save=True, img_csv=False):
    truth_lon = lon
    truth_lat = lat
    fire_lon_index = lon_list.index(truth_lon)
    fire_lat_index = lat_list.index(truth_lat)

    folder_path = 'fire_detection/img/' + truth_time  # 创建文件夹保存各点通道信息
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    lat_start = int()
    lat_end = int()
    lon_start = int()
    lon_end = int()

    pixel_distance = 7

    if fire_lat_index < pixel_distance:
        lat_start = 0
        lat_end = fire_lat_index + pixel_distance + 1
    elif pixel_distance <= fire_lat_index < len(lat_list) - pixel_distance:
        lat_start = fire_lat_index - pixel_distance
        lat_end = fire_lat_index + pixel_distance + 1
    else:
        lat_start = fire_lat_index - pixel_distance
        lat_end = 1001

    if fire_lon_index < pixel_distance:
        lon_start = 0
        lon_end = fire_lon_index + pixel_distance + 1
    elif pixel_distance <= fire_lon_index < len(lon_list) - pixel_distance:
        lon_start = fire_lon_index - pixel_distance
        lon_end = fire_lon_index + pixel_distance + 1
    else:
        lon_start = fire_lon_index - pixel_distance
        lon_end = 1101

    fire_matrix_index = [fire_lat_index - lat_start, fire_lon_index - lon_start]

    if img_save:
        misc.imsave(folder_path + '/band2' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg',
                    h8_band2_data[lat_start:lat_end, lon_start:lon_end])
        pd.DataFrame(h8_band2_data[lat_start:lat_end, lon_start:lon_end]).to_csv(
            folder_path + '/band2' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.csv')

        misc.imsave(folder_path + '/band3' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg',
                    h8_band3_data[lat_start:lat_end, lon_start:lon_end])
        pd.DataFrame(h8_band3_data[lat_start:lat_end, lon_start:lon_end]).to_csv(
            folder_path + '/band3' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.csv')

        misc.imsave(folder_path + '/band4' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg',
                    h8_band4_data[lat_start:lat_end, lon_start:lon_end])
        pd.DataFrame(h8_band4_data[lat_start:lat_end, lon_start:lon_end]).to_csv(
            folder_path + '/band4' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.csv')

        misc.imsave(folder_path + '/band7' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg',
                    h8_band7_data[lat_start:lat_end, lon_start:lon_end])
        pd.DataFrame(h8_band7_data[lat_start:lat_end, lon_start:lon_end]).to_csv(
            folder_path + '/band7' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.csv')

        misc.imsave(folder_path + '/band7_14' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.jpg',
                    h8_band7_14_data[lat_start:lat_end, lon_start:lon_end])
        pd.DataFrame(h8_band7_14_data[lat_start:lat_end, lon_start:lon_end]).to_csv(
            folder_path + '/band7_14' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.csv')

        imgProcess(folder_path, fire_matrix_index, truth_lon, truth_lat)
    if img_csv:
        pd.DataFrame(h8_band2_data[lat_start:lat_end, lon_start:lon_end]).to_csv(
            folder_path + '/band2' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.csv')
        pd.DataFrame(h8_band3_data[lat_start:lat_end, lon_start:lon_end]).to_csv(
            folder_path + '/band3' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.csv')
        pd.DataFrame(h8_band4_data[lat_start:lat_end, lon_start:lon_end]).to_csv(
            folder_path + '/band4' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.csv')
        pd.DataFrame(h8_band7_data[lat_start:lat_end, lon_start:lon_end]).to_csv(
            folder_path + '/band7' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.csv')
        pd.DataFrame(h8_band7_14_data[lat_start:lat_end, lon_start:lon_end]).to_csv(
            folder_path + '/band7_14' + '_' + str(truth_lon) + '_' + str(truth_lat) + '.csv')


def cloudEliminate(file, threshold):
    path = './fire_detection/output'
    output_temp = pd.read_csv(path + '/' + file)
    if output_temp.shape[0] > 0 and len(output_temp.columns.tolist()) > 1:
        fire_lons = output_temp['Lons'].values.reshape(-1)
        fire_lats = output_temp['Lats'].values.reshape(-1)
        if len(fire_lons) >= 2 or len(fire_lats) >= 2:
            comb_lons = list()
            comb_lats = list()

            resize_ptrs = np.array([[item_x, item_y] for item_x, item_y in zip(fire_lons, fire_lats)])
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
            output_dbscan['Time'] = output_temp['Time'][0]
            time_str = output_temp['Time'][0]
            time_str = datetime.datetime.strftime(datetime.datetime.strptime(time_str, "%Y/%m/%d %H:%M"), '%Y%m%d%H%M')
            output_dbscan.to_csv('./fire_detection/output_dbscan/' + 'SHJC' + time_str + '_dbscan.csv')
            output_temp['dbscan'] = ptrs_clusters.labels_
        else:
            output_temp['dbscan'] = -1
        lables = np.unique(output_temp['dbscan'].values).tolist()
        output_temp_new = pd.DataFrame()
        for lable in lables:
            block = output_temp[output_temp['dbscan'] == lable]         # 选取output_temp中所有标签为label的数据
            if block['n_cloud'].max() < 9 and block['n_cloud_band2'].max() < 10:        # n_cloud - 通道3减通道4反射率标准化结果; n_cloud_band2 - 通道2反射率标准化结果
                if block.shape[0] < threshold:
                    output_temp_new = pd.concat([output_temp_new, block], axis=0)       # 如果当前聚类中火点数量小于阈值则将output_temp中火点加入output_temp_new中
        if 'Lons' in output_temp_new.columns.tolist():
            output_temp_new.to_csv('./fire_detection/output_eliminate/' + file)


# 对POI_time时间的影像pk文件，做火点检测
def POI_to_params(POI_time, argsDict):
    fire_absolute_df = pd.DataFrame()       # 存储绝对火点(绝对火点判断方式：7通道亮温足够大，前后两帧影像的7通道增量足够大，7-14通道增量足够大)
    result_variable_fire = pd.DataFrame()   # 存储火点像元，以及像元的相关特征

    # 从argsDict中获取多组阈值
    n3_th = argsDict['n3']
    n4_th = argsDict['n4']
    n5_th = argsDict['n5']
    b3plusb4_th = argsDict['b3plusb4']
    band7_th = argsDict['band7_th']
    band7_incre_th = argsDict['band7_incre']
    band7_14_incre_th = argsDict['band7_14_incre']
    n2_th_night = argsDict['n2_night']
    cluster_limit = argsDict['cluster_limit']
    # 上一时刻，即十分钟前，用于进行前后2帧影像变化的绝对火点判定
    truth_time = POI_time.strftime('%Y%m%d%H%M')
    truth_time_last = POI_time - datetime.timedelta(seconds=0, minutes=10, hours=0)
    truth_time_last = truth_time_last.strftime('%Y%m%d%H%M')
    # pk文件存储在特定文件夹
    pk_path = './fire_detection/H8_pk/SHJC' + truth_time + '.pk'
    pk_path_last = 'fire_detection/H8_pk/SHJC' + truth_time_last + '.pk'
    order = ['Time', 'Lon', 'Lat', 'cloud?', 'water?', 'night?', 'n1_std_valid', 'n2_std_valid', 'n3', 'n4', 'n5',
             'band7_deta',
             'band7-14_deta', 'band7-14_mean_size7', 'n2_std_valid_size21', 'band3+4']
    fire_result_df = pd.DataFrame(columns=order)
    if not os.path.exists(pk_path):
        fire_result_df = pd.DataFrame(columns=order)
        return fire_result_df
    else:
        # 从pk文件中，读取经纬度，和16个波段的影像
        h8_f = open(pk_path, 'rb')
        h8_data = pickle.load(h8_f)
        lon = np.array(h8_data[0])  # 经度
        lat = np.array(h8_data[1])  # 纬度
        h8_band2_data = h8_data[3]
        h8_band3_data = h8_data[4]  # 通道3的卫星影像：1000*1100的矩阵
        h8_band4_data = h8_data[5]
        h8_band6_data = h8_data[7]
        h8_band7_data = h8_data[8]
        h8_band14_data = h8_data[15]
        h8_band15_data = h8_data[16]
        h8_band7_14_data = h8_band7_data - h8_band14_data
        h8_band34_data = h8_band3_data + h8_band4_data
        lon_list = [round(item, 2) for item in list(lon)]
        lat_list = [round(item, 2) for item in list(lat)]
        # 云掩膜
        cloud_img = h8_band3_data.copy()
        for lats_index in range(len(lat)):
            for lons_index in range(len(lon)):
                nightBool = abs(h8_band3_data[lats_index, lons_index]) < 0.01 and \
                            abs(h8_band4_data[lats_index, lons_index]) < 0.01
                nonCloudBool = (h8_band3_data[lats_index, lons_index] + h8_band4_data[lats_index, lons_index] < 0.36) \
                            and (h8_band15_data[lats_index, lons_index] > 265) \
                            and ((h8_band3_data[lats_index, lons_index] + h8_band4_data[lats_index, lons_index] < 0.32)
                            or (h8_band15_data[lats_index, lons_index] > 285)) \
                            or nightBool
                if nonCloudBool:
                    cloud_img[lats_index, lons_index] = 0  # 非云
                else:
                    cloud_img[lats_index, lons_index] = 1  # 云
        # 读取上一帧的pk影像，用于绝对火点判定
        if os.path.exists(pk_path_last):
            with open(pk_path_last, 'rb') as h8_f_last:
                h8_data_last = pickle.load(h8_f_last)
                h8_band7_data_last = h8_data_last[8]
                h8_band14_data_last = h8_data_last[15]
                h8_band7_14_data_last = h8_band7_data_last - h8_band14_data_last
                band7_deta2 = h8_band7_data - h8_band7_data_last
                band7_14_deta2 = h8_band7_14_data - h8_band7_14_data_last

        i = 0  # 针对每一个像素做火点判识，i用于记录循环次数
        yunnan_min_lon_index = lon_list.index(97.50)
        yunnan_max_lon_index = lon_list.index(106.20)
        yunnan_min_lat_index = lat_list.index(21.12)
        yunnan_max_lat_index = lat_list.index(29.26)
        pixel_num = ((106.20 - 97.50) / 0.01 + 1) * ((29.26 - 21.12) / 0.01 + 1)
        print('processing: ' + 'SHJC' + truth_time + '.pk')
        # 将搜索区域缩小至云南范围内
        for fire_lon_index in list(range(len(lon_list)))[yunnan_min_lon_index:yunnan_max_lon_index + 1]:
            for fire_lat_index in list(range(len(lat_list)))[yunnan_max_lat_index:yunnan_min_lat_index + 1]:
                i = i + 1
                if i % 50000 == 0:
                    print("\rProgress: " + str(round(i * 100 / pixel_num, 2)) + "%", end="")
                if os.path.exists(pk_path_last):
                    band7_deta = band7_deta2[fire_lat_index, fire_lon_index]
                    band7_14_deta = band7_14_deta2[fire_lat_index, fire_lon_index]
                else:
                    band7_deta = -999
                    band7_14_deta = -999
                # 若该像元为云像元，跳过
                if cloud_img[fire_lat_index, fire_lon_index] == 1:
                    continue
                # 该像元背景像元的云像元比例超过25%，直接判断为非火点(有效像元不足70%)
                size = 15
                kernal_point = int((size - 1) / 2)
                x_start = fire_lat_index - kernal_point
                y_start = fire_lon_index - kernal_point
                kernel_cloud = cloud_img[x_start:x_start + size, y_start:y_start + size]
                if np.sum(kernel_cloud) / (15 * 15) >= 0.3:
                    continue
                # 云边检测
                kernal_point = int((5 - 1) / 2)
                x_start = fire_lat_index - kernal_point
                y_start = fire_lon_index - kernal_point
                kernel_cloud_edge = cloud_img[x_start:x_start + 5, y_start:y_start + 5]
                if np.sum(kernel_cloud_edge) > 6:
                    continue
                # 排除7通道亮温值过低的火点
                if h8_band7_data[fire_lat_index, fire_lon_index] <= 260:
                    continue
                # 计算像元的若干特征，用于火点判别
                # n3 - 通道3反射率; n4 - 通道7亮温值; n5 - 通道7减通道14亮温差值;
                # n1_std_with_valid_pixels - 通道7亮温值标准化结果; n2_std_with_valid_pixels - 通道7减通道14亮温差值标准化结果;
                # band7_14_mean - 通道7减通道14亮温均值; nightBool - 夜判结果; cloudBool - 云判结果; waterBool - 水判结果;
                # band7_mean_size7 - 小窗口通道7均值; band7_14_mean_size7 - 小窗口通道7减通道14亮温均值;
                # n_cloud - 通道3减通道4反射率标准化结果, n_cloud_band2 - 通道2反射率标准化结果
                n3, n4, n5, n1_std_with_valid_pixels, n2_std_with_valid_pixels, band7_14_mean, nightBool, cloudBool, waterBool,\
                band7_mean_size7, band7_14_mean_size7, n_cloud, n_cloud_band2 = fdj.hd_test(h8_band3_data,
                                                                                            h8_band7_data,
                                                                                            h8_band14_data,
                                                                                            fire_lat_index,
                                                                                            fire_lon_index,
                                                                                            h8_band4_data,
                                                                                            h8_band15_data,
                                                                                            h8_band6_data,
                                                                                            h8_band34_data,
                                                                                            h8_band2_data, kernel_cloud,
                                                                                            size=15)
                band3_data = h8_band3_data[fire_lat_index, fire_lon_index]  # 获取3通道反射率
                band4_data = h8_band4_data[fire_lat_index, fire_lon_index]  # 获取4通道反射率
                band3plus4_data = band3_data + band4_data  # 计算通道3与通道4反射率的和，用于判云
                datetime_tmp = POI_time.strftime("%Y/%m/%d %H:%M")
                # 绝对火点判据
                if n4 > band7_th:
                    fire_absolute_df = fire_absolute_df.append(
                        {'Time': datetime_tmp, 'Lons': lon_list[fire_lon_index], 'Lats': lat_list[fire_lat_index],
                         'band7': n4, 'band7_deta': band7_deta, 'band7-14_deta': band7_14_deta}, ignore_index=True)
                    continue
                elif band7_deta > band7_incre_th and band7_14_deta > band7_14_incre_th:
                    fire_absolute_df = fire_absolute_df.append(
                        {'Time': datetime_tmp, 'Lons': lon_list[fire_lon_index], 'Lats': lat_list[fire_lat_index],
                         'band7': n4, 'band7_deta': band7_deta, 'band7-14_deta': band7_14_deta}, ignore_index=True)
                    continue
                # 基于背景像元的相对火点判据
                n1plusn2 = n1_std_with_valid_pixels + n2_std_with_valid_pixels
                if 6.34 <= n1plusn2 < 7 and (n1plusn2 + band7_14_mean_size7 > 14) and n1_std_with_valid_pixels > 2.91 and n5 > 12.3:
                    result_variable_fire = result_variable_fire.append(
                        {'Time': datetime_tmp, 'Lons': lon_list[fire_lon_index], 'Lats': lat_list[fire_lat_index],
                         'cloud?': cloudBool, 'water?': waterBool, 'night?': nightBool,
                         'n1_std_valid': n1_std_with_valid_pixels, 'n2_std_valid': n2_std_with_valid_pixels, 'n3': n3,
                         'n4': n4, 'n5': n5, 'n1+n2': n1plusn2, 'band7_deta': band7_deta,
                         'band7-14_deta': band7_14_deta, 'band7_mean_size7': band7_mean_size7,
                         'band7-14_mean_size7': band7_14_mean_size7,
                         'band3+4': band3plus4_data, 'n_cloud': n_cloud, 'n_cloud_band2': n_cloud_band2},
                        ignore_index=True)
                    continue
                if band7_14_mean_size7 > 6.57 and 6.35 < n1plusn2 < 7.0 and n1_std_with_valid_pixels > 2.91 and n5 > 12.3:
                    result_variable_fire = result_variable_fire.append(
                        {'Time': datetime_tmp, 'Lons': lon_list[fire_lon_index], 'Lats': lat_list[fire_lat_index],
                         'cloud?': cloudBool, 'water?': waterBool, 'night?': nightBool,
                         'n1_std_valid': n1_std_with_valid_pixels, 'n2_std_valid': n2_std_with_valid_pixels, 'n3': n3,
                         'n4': n4, 'n5': n5, 'n1+n2': n1plusn2, 'band7_deta': band7_deta,
                         'band7-14_deta': band7_14_deta, 'band7_mean_size7': band7_mean_size7,
                         'band7-14_mean_size7': band7_14_mean_size7,
                         'band3+4': band3plus4_data, 'n_cloud': n_cloud, 'n_cloud_band2': n_cloud_band2},
                        ignore_index=True)
                    continue
                if 7.0 <= n1plusn2 < 8 and n1plusn2 + band7_14_mean_size7 > 13.4 and n1_std_with_valid_pixels > 2.42 and n5 > 10.2:
                    result_variable_fire = result_variable_fire.append(
                        {'Time': datetime_tmp, 'Lons': lon_list[fire_lon_index], 'Lats': lat_list[fire_lat_index],
                         'cloud?': cloudBool, 'water?': waterBool, 'night?': nightBool,
                         'n1_std_valid': n1_std_with_valid_pixels, 'n2_std_valid': n2_std_with_valid_pixels, 'n3': n3,
                         'n4': n4, 'n5': n5, 'n1+n2': n1plusn2, 'band7_deta': band7_deta,
                         'band7-14_deta': band7_14_deta, 'band7_mean_size7': band7_mean_size7,
                         'band7-14_mean_size7': band7_14_mean_size7,
                         'band3+4': band3plus4_data, 'n_cloud': n_cloud, 'n_cloud_band2': n_cloud_band2},
                        ignore_index=True)
                    continue
                if band7_14_mean_size7 > 4.4 and 7.61 < n1plusn2 < 8.0 and n1_std_with_valid_pixels > 3.47 and n5 > 8.4:
                    result_variable_fire = result_variable_fire.append(
                        {'Time': datetime_tmp, 'Lons': lon_list[fire_lon_index], 'Lats': lat_list[fire_lat_index],
                         'cloud?': cloudBool, 'water?': waterBool, 'night?': nightBool,
                         'n1_std_valid': n1_std_with_valid_pixels, 'n2_std_valid': n2_std_with_valid_pixels, 'n3': n3,
                         'n4': n4, 'n5': n5, 'n1+n2': n1plusn2, 'band7_deta': band7_deta,
                         'band7-14_deta': band7_14_deta, 'band7_mean_size7': band7_mean_size7,
                         'band7-14_mean_size7': band7_14_mean_size7,
                         'band3+4': band3plus4_data, 'n_cloud': n_cloud, 'n_cloud_band2': n_cloud_band2},
                        ignore_index=True)
                    continue
                if 8.0 <= n1plusn2 < 9 and n1plusn2 + band7_14_mean_size7 > 12.38 and n1_std_with_valid_pixels > 2.42 and n5 > 8.8:
                    result_variable_fire = result_variable_fire.append(
                        {'Time': datetime_tmp, 'Lons': lon_list[fire_lon_index], 'Lats': lat_list[fire_lat_index],
                         'cloud?': cloudBool, 'water?': waterBool, 'night?': nightBool,
                         'n1_std_valid': n1_std_with_valid_pixels, 'n2_std_valid': n2_std_with_valid_pixels, 'n3': n3,
                         'n4': n4, 'n5': n5, 'n1+n2': n1plusn2, 'band7_deta': band7_deta,
                         'band7-14_deta': band7_14_deta, 'band7_mean_size7': band7_mean_size7,
                         'band7-14_mean_size7': band7_14_mean_size7,
                         'band3+4': band3plus4_data, 'n_cloud': n_cloud, 'n_cloud_band2': n_cloud_band2},
                        ignore_index=True)
                    continue
                if band7_14_mean_size7 > 2.1 and 8 < n1plusn2 < 9.0 and n2_std_with_valid_pixels > 4.26 and n5 > 6.6:
                    result_variable_fire = result_variable_fire.append(
                        {'Time': datetime_tmp, 'Lons': lon_list[fire_lon_index], 'Lats': lat_list[fire_lat_index],
                         'cloud?': cloudBool, 'water?': waterBool, 'night?': nightBool,
                         'n1_std_valid': n1_std_with_valid_pixels, 'n2_std_valid': n2_std_with_valid_pixels, 'n3': n3,
                         'n4': n4, 'n5': n5, 'n1+n2': n1plusn2, 'band7_deta': band7_deta,
                         'band7-14_deta': band7_14_deta, 'band7_mean_size7': band7_mean_size7,
                         'band7-14_mean_size7': band7_14_mean_size7,
                         'band3+4': band3plus4_data, 'n_cloud': n_cloud, 'n_cloud_band2': n_cloud_band2},
                        ignore_index=True)
                    continue
                if 9.0 <= n1plusn2 < 10.0 and n1plusn2 + band7_14_mean_size7 > 10.4 and n1_std_with_valid_pixels > 4 and n5 > 7.1:
                    result_variable_fire = result_variable_fire.append(
                        {'Time': datetime_tmp, 'Lons': lon_list[fire_lon_index], 'Lats': lat_list[fire_lat_index],
                         'cloud?': cloudBool, 'water?': waterBool, 'night?': nightBool,
                         'n1_std_valid': n1_std_with_valid_pixels, 'n2_std_valid': n2_std_with_valid_pixels, 'n3': n3,
                         'n4': n4, 'n5': n5, 'n1+n2': n1plusn2, 'band7_deta': band7_deta,
                         'band7-14_deta': band7_14_deta, 'band7_mean_size7': band7_mean_size7,
                         'band7-14_mean_size7': band7_14_mean_size7,
                         'band3+4': band3plus4_data, 'n_cloud': n_cloud, 'n_cloud_band2': n_cloud_band2},
                        ignore_index=True)
                    continue
                if n1plusn2 >= 10.0 and n1plusn2 + band7_14_mean_size7 > 0.8 and n1_std_with_valid_pixels > 4.35 and n5 > 6.0:
                    result_variable_fire = result_variable_fire.append(
                        {'Time': datetime_tmp, 'Lons': lon_list[fire_lon_index], 'Lats': lat_list[fire_lat_index],
                         'cloud?': cloudBool, 'water?': waterBool, 'night?': nightBool,
                         'n1_std_valid': n1_std_with_valid_pixels, 'n2_std_valid': n2_std_with_valid_pixels, 'n3': n3,
                         'n4': n4, 'n5': n5, 'n1+n2': n1plusn2, 'band7_deta': band7_deta,
                         'band7-14_deta': band7_14_deta, 'band7_mean_size7': band7_mean_size7,
                         'band7-14_mean_size7': band7_14_mean_size7,
                         'band3+4': band3plus4_data, 'n_cloud': n_cloud, 'n_cloud_band2': n_cloud_band2},
                        ignore_index=True)
                    continue
                # 夜晚的火点判据
                if nightBool and n2_std_with_valid_pixels > n2_th_night and n3 < n3_th and band3plus4_data < b3plusb4_th and n4 > n4_th and n5 > n5_th:
                    result_variable_fire = result_variable_fire.append(
                        {'Time': datetime_tmp, 'Lons': lon_list[fire_lon_index], 'Lats': lat_list[fire_lat_index],
                         'cloud?': cloudBool, 'water?': waterBool, 'night?': nightBool,
                         'n1_std_valid': n1_std_with_valid_pixels, 'n2_std_valid': n2_std_with_valid_pixels, 'n3': n3,
                         'n4': n4, 'n5': n5, 'n1+n2': n1plusn2,
                         'band7_deta': band7_deta, 'band7-14_deta': band7_14_deta, 'band7_mean_size7': band7_mean_size7,
                         'band7-14_mean_size7': band7_14_mean_size7, 'band3+4': band3plus4_data,
                         'n_cloud': n_cloud, 'n_cloud_band2': n_cloud_band2},
                        ignore_index=True)
        print("\n")     # 格式化输出数据
        csv_name = 'SHJC' + POI_time.strftime('%Y%m%d%H%M') + '5.csv'
        # DBSCAN聚类
        cluster_flag = True
        output_temp = result_variable_fire.copy()
        output_dbscan = pd.DataFrame()
        output_dbscan.to_csv('./fire_detection/output_dbscan/' + 'SHJC' + POI_time.strftime('%Y%m%d%H%M') + '_dbscan.csv')
        if cluster_flag and output_temp.shape[0] > 1:
            fire_lons = output_temp['Lons'].values.reshape(-1)
            fire_lats = output_temp['Lats'].values.reshape(-1)
            if len(fire_lons) >= 2 or len(fire_lats) >= 2:
                comb_lons = list()
                comb_lats = list()

                resize_ptrs = np.array([[item_x, item_y] for item_x, item_y in zip(fire_lons, fire_lats)])
                ptrs_clusters = DBSCAN(eps=0.03, min_samples=2).fit(resize_ptrs)
                max_cluster_label = np.max(ptrs_clusters.labels_)
                for cluster_i in np.arange(0, max_cluster_label + 1):
                    cur_clust_elmidxs = np.where(ptrs_clusters.labels_ == cluster_i)[0]     # 获取当前便签为i的聚类元素索引值
                    cur_clust_avglon = np.mean(resize_ptrs[cur_clust_elmidxs, 0])           # 计算当前聚类平均经度
                    cur_clust_avglat = np.mean(resize_ptrs[cur_clust_elmidxs, 1])           # 计算当前聚类平均纬度
                    comb_lons.append(round(cur_clust_avglon, 2))
                    comb_lats.append(round(cur_clust_avglat, 2))

                fire_lons_dbscan = comb_lons
                fire_lats_dbscan = comb_lats
                output_dbscan = pd.DataFrame()
                output_dbscan['Lons'] = fire_lons_dbscan
                output_dbscan['Lats'] = fire_lats_dbscan
                output_dbscan['Time'] = output_temp['Time'][0]
                time_str = output_temp['Time'][0]
                time_str = datetime.datetime.strftime(datetime.datetime.strptime(time_str, "%Y/%m/%d %H:%M"), '%Y%m%d%H%M')
                output_dbscan.to_csv('./fire_detection/output_dbscan/' + 'SHJC' + time_str + '_dbscan.csv')
                output_temp['dbscan'] = ptrs_clusters.labels_
            # 针对聚类结果，生成带标记的火点图像
            for i in range(output_dbscan.shape[0]):
                lon_db_temp = output_dbscan.iloc[i, 0]
                lat_db_temp = output_dbscan.iloc[i, 1]
                folder_path = 'fire_detection/img/' + time_str
                imgGeneration(folder_path, lon_db_temp, lat_db_temp, lon_list, lat_list, time_str, h8_band2_data,
                              h8_band3_data, h8_band4_data, h8_band7_data, h8_band7_14_data, img_save=True, img_csv=False)
        output_temp.to_csv("./fire_detection/output/" + csv_name, index=False)
        # 降低虚警率，进一步剔除局部云遮挡的像元，对于识别的火点聚类，若面积过大则不是火
        if output_temp.shape[0] > 0:
            cloudEliminate(csv_name, cluster_limit)
        csv_name_absolute = 'SHJC' + POI_time.strftime('%Y%m%d%H%M') + '5_absolute.csv'
        fire_absolute_df.to_csv("./fire_detection/output_absolute/" + csv_name_absolute)        # 将绝对火点保存至output_absolute文件夹下


# params算法参数
def fire_detection(params):
    if not (os.path.exists('./fire_detection/H8_pk')):
        print("ERROR: can't find the path: ./fire_detection/H8_pk")
        sys.exit(-3)
    H8_files = os.listdir('./fire_detection/H8_pk')
    output_files = os.listdir('./fire_detection/output')
    time_list = [item.split(".")[0][4:] for item in H8_files if item.endswith(".pk")]
    output_files_times = [item.split(".")[0][4:16] for item in output_files if item.endswith(".csv")]
    # 对未计算过的火点像元重新计算
    for time_curr in time_list:
        if time_curr not in output_files_times:
            POI_time = datetime.datetime.strptime(time_curr, '%Y%m%d%H%M')
            POI_to_params(POI_time, params)


def main():
    # 山火识别特征参数阈值
    argsDict = {"n1": 1.8,
                "n2": 2.0,      # 7通道，15*15窗口下，窗口中心亮温值与背景像元亮温的差异的阈值(差异越大，越可能是火，n1调高，会增加火点数量)
                "n3": 0.13,     # 3通道阈值，可见光(云会反光，因此，3通道越小，越不可能是云)
                "n4": 283,      # 7通道阈值，中红外 (7通道，即中红外的亮温值越高，越可能是火)
                "n5": 3.3,      # 7通道-14通道 阈值，中红外-远红外
                "b3plusb4": 0.3415,     # 3通道+4通道阈值，2个可见光通道的和，用于判云
                "band7_th": 330,        # 当7通道的亮温大于band7_th时，为绝对火点
                "n2_size21": 2.9379,    # 废弃
                "band7_incre": 20,      # 用于判定绝对火电，前后2帧影像的7通道的亮温值增量的阈值
                "band7_14_incre": 20,   # 用于判定绝对火电，前后2帧影像的7通道减去14通道的亮温值增量的阈值
                "band7_14_mean": 20,    # 废弃，15*15窗口下，7-14通道亮温值的均值
                "band7_mean_size7": 302,        # 考虑废弃，7*7窗口下，7通道亮温值的均值，用于判定大火；目前看效果一般，后续考虑废弃
                "band7_14_mean_size7": 6.3,     # 考虑废弃，7*7窗口下，7通道-14通道亮温值的均值，用于判定大火；目前看效果一般，后续考虑废弃
                "band7_14_mean_size7_2": 15,  # 30
                "n1+n2": 5.5,       # 7通道背景差异和7-14通道背景差异相加的阈值，有些情况下，7通道背景差异或7-14通道背景差异，两者之一非常大，另一个较小，也是火
                "n1_2": 3.2,                    # 废弃
                "n2_2": 1.0,                    # 废弃
                "n2_night": 4.39,               # 夜间判别
                "cluster_limit": 20             # 一片火的像素个数不超过cluster_limit

                }
    # 山火检测
    fire_detection(argsDict)
    # 处理临时文件
    filelist_output = os.listdir('./fire_detection/output/')
    filelist_eliminate = os.listdir('./fire_detection/output_eliminate/')
    for fileTemp in filelist_output:
        if fileTemp not in filelist_eliminate:
            order = ['Time', 'Lons', 'Lats', 'cloud?', 'water?', 'night?', 'n1_std_valid', 'n2_std_valid', 'n3', 'n4',
                     'n5', 'n1+n2', 'band7_deta', 'band7-4_deta', 'band7_mean_size7', 'band7-14_mean_size7', 'band3+4',
                     'n_cloud', 'n_cloud_band2']
            file_csv = pd.DataFrame(columns=order)
            file_csv.to_csv('./fire_detection/output_eliminate/' + fileTemp)


if __name__ == '__main__':
    main()
