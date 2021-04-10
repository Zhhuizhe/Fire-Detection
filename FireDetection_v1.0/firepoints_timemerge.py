import glob
import datetime
import requests
import json
import numpy as np
import pandas as pd
import pickle as pkl
from database import Database
from himawari import Himawari
from os import path
from sklearn.cluster import DBSCAN
from scipy.spatial import distance as dist


def get_merge_csvdata(firetimes, firecsvpaths, out_csv_datetime, mergetime):
    merge_stop_time = out_csv_datetime
    merge_start_time = out_csv_datetime - datetime.timedelta(seconds=mergetime)
    within_mergetime_bools = [merge_start_time <= fire_time <= merge_stop_time for fire_time in firetimes]
    within_mergetime_csvpaths = [firecsvpaths[idx] for idx in range(len(firecsvpaths)) if within_mergetime_bools[idx]]
    within_mergetime_datetimes = [firetimes[idx] for idx in range(len(firecsvpaths)) if within_mergetime_bools[idx]]

    if len(within_mergetime_csvpaths) <= 0:
        return np.array([]), np.array([]), None

    fire_points_arr_ = list()
    for file_path, fire_time in zip(within_mergetime_csvpaths, within_mergetime_datetimes):
        file_path = file_path.strip()
        cur_encoding = 'GB2312'
        csv_fdata = pd.read_csv(file_path, encoding=cur_encoding)
        cur_csv_lons = csv_fdata[r'Lons'].values.flatten()
        cur_csv_lats = csv_fdata[r'Lats'].values.flatten()
        # 将融合时间范围内的csv文件火点经纬度保存在fire_points_arr_列表中
        fire_points = list()
        for lon, lat in zip(cur_csv_lons, cur_csv_lats):
            fire_points.append([lon, lat])
        fire_points_arr_.append(np.array(fire_points))
    fire_times_ = np.array(within_mergetime_datetimes)
    fire_points_arr_ = np.array(fire_points_arr_, dtype=object)
    indices = np.argsort(fire_times_)
    max_time = fire_times_[indices[-1]]
    fire_times_ = (max_time - fire_times_)      # 得到时间差值列表
    fire_times = np.array([fire_times_[idx].total_seconds() * -1 for idx in indices])       # 将时间差以秒为单位，按负值的形式存储在列表中
    fire_points_arr = fire_points_arr_[indices]         # 将火点列表按相同顺序存储在fire_points_arr中
    # 将列表中融合时间内的火点保存在fire_points_list中
    fire_times_list = list()
    fire_points_list = list()
    for fire_time, fire_points in zip(fire_times, fire_points_arr):
        if np.abs(fire_time) <= mergetime:
            fire_times_list.append(fire_time)
            fire_points_list.append(fire_points)

    return np.array(fire_times_list), np.array(fire_points_list, dtype=object), within_mergetime_csvpaths[indices[-1]]


def merge_pair_firepoints(fire_points, merge_rng, min_samples):
    fire_points_array = np.vstack([item for item in fire_points if len(item) > 0])
    db_obj = DBSCAN(eps=merge_rng, min_samples=min_samples).fit(fire_points_array)
    clust_ptrlbls = db_obj.labels_

    # 保留超过1个点的聚类结果
    uniq_clust_ptrlbls = np.unique(clust_ptrlbls)
    merge_point_idxs = list()

    for clust_lbl in uniq_clust_ptrlbls:
        cur_clust_idxs = np.where(clust_ptrlbls == clust_lbl)[0]
        cur_clust_nelms = len(cur_clust_idxs)
        # 摒弃只有单点的聚类都不要了
        if cur_clust_nelms > 1:
            if (fire_points_array[cur_clust_idxs[0], 0] - fire_points_array[cur_clust_idxs[-1], 0]) > 0.023 or \
                    (fire_points_array[cur_clust_idxs[0], 1] - fire_points_array[cur_clust_idxs[-1], 1] > 0.023):
                merge_point_idxs.append(cur_clust_idxs[-1])
            else:
                merge_point_idxs.append(cur_clust_idxs[-1])
    merge_points = [(fire_points_array[idx, 0], fire_points_array[idx, 1]) for idx in merge_point_idxs]
    merge_points = np.array(merge_points)
    return merge_points


# 读取植被信息
def read_plant_info(Xgeo, Ygeo, data_dict):
    GT = data_dict['GT']
    data = data_dict['data']

    dTemp = GT[1]*GT[5]-GT[2]*GT[4]
    dcol = (GT[5]*(Xgeo-GT[0])-GT[2]*(Ygeo-GT[3]))/dTemp+0.5
    drow = (GT[1]*(Ygeo-GT[3])-GT[4]*(Xgeo-GT[0]))/dTemp+0.5
    icol = int(dcol)
    irow = int(drow)
    forestland = 0
    grassland = 0
    farmland = 0
    other = 0

    for i in range(4):
        for j in range(4):
            if data[irow - 1 + i, icol - 1 + j] < 6:
                forestland += 1
            elif 5 < data[irow - 1 + i, icol - 1 + j] < 11:
                grassland += 1
            elif 10 < data[irow - 1 + i, icol - 1 + j] < 16:
                farmland += 1
            else:
                other += 1

    return int(forestland/0.16), int(grassland/0.16), int(farmland/0.16), int(other/0.16)


# 筛除固定火点
def filter_stable_heatptrs(stable_hp_xls, fire_lons, fire_lats):
    stable_hp_filtrng = 0.007
    stable_hp_df = pd.read_excel(stable_hp_xls)
    stable_hp_lons = stable_hp_df[r'经度'].values
    stable_hp_lats = stable_hp_df[r'纬度'].values
    stable_hp_lons = np.array([lon_val for lon_val in stable_hp_lons if not np.isnan(lon_val)])
    stable_hp_lats = np.array([lat_val for lat_val in stable_hp_lats if not np.isnan(lat_val)])
    stable_hp_llts = np.stack([stable_hp_lons, stable_hp_lats]).T
    cur_pred_llts = np.stack([np.array(fire_lons), np.array(fire_lats)]).T

    pred2shp_dists = dist.cdist(cur_pred_llts, stable_hp_llts)
    pred2shp_min_dists = pred2shp_dists.min(axis=1)
    predfp_b_stablehp = pred2shp_min_dists <= stable_hp_filtrng

    fire_lons = [fire_lon for fire_lon, b_shp in zip(fire_lons, predfp_b_stablehp) if not b_shp]
    fire_lats = [fire_lat for fire_lat, b_shp in zip(fire_lats, predfp_b_stablehp) if not b_shp]

    return fire_lons, fire_lats


# 市县查询
def info_search(longitude, latitude):
    try:
        url = 'https://restapi.amap.com/v3/geocode/regeo'
        params = {'key': 'c9c5ca2a25f6e9940adeb39f7b6e9c4f', 'location': str(longitude) + "," + str(latitude)}
        res = requests.get(url, params)
        js = json.loads(res.text)
        city = str(js["regeocode"]["addressComponent"]["city"]) if js["regeocode"]["addressComponent"]["city"] else ""
        district = str(js["regeocode"]["addressComponent"]["district"]) if js["regeocode"]["addressComponent"]["district"] else ""
        return city, district
    except Exception:
        return


def fire_timemerge(file_list, pk_src_path, params):
    mergetime = params["火点融合时间"]        # 火点融合时间差
    merge_rng = params["火点融合经纬差值"]        # 火点融合经纬差值
    min_samples = 1
    merge_path = './fire_detection/output_eliminate/'
    pk_path = './fire_detection/H8_pk/'
    for filename in file_list:
        if not path.exists(merge_path + filename):
            continue
        output_csv_datetime = datetime.datetime.strptime(filename[4:16], '%Y%m%d%H%M')
        all_fire_csvs = glob.glob(path.join(merge_path, '*.csv'))          # 遍历output_eliminate文件夹下所有csv文件
        all_fire_times = list()
        # 遍历eliminate文件夹下所有csv文件，将文件名中的日期信息提取至all_fire_times列表中
        for file_path in all_fire_csvs:
            file_path = file_path.strip()
            if not path.isfile(file_path):
                continue
            file_name = path.splitext(path.basename(file_path))[0]
            file_name = file_name.replace('SHJC', '')
            file_name = file_name[:len(file_name) - 1]

            if len(file_name) == 12:
                fire_year = int(file_name[0:4])
                fire_month = int(file_name[4:6])
                fire_day = int(file_name[6:8])
                fire_hour = int(file_name[8:10])
                fire_minute = int(file_name[10:12])
                fire_time = datetime.datetime(fire_year, fire_month, fire_day, fire_hour, fire_minute)
                all_fire_times.append(fire_time)
            else:
                continue

        if len(all_fire_times) <= 0:
            print("ERROR:can't find enough files in the last %d seconds" % mergetime)
            continue

        fire_times, fire_points_arr, last_fname = get_merge_csvdata(all_fire_times, all_fire_csvs, output_csv_datetime, mergetime)
        # 若在融合时间范围内没有找到csv文件，则跳过
        if len(fire_times) <= 1:
            continue
        latenc_asc_idxs = np.argsort(fire_times)[::-1]
        fire_lons = list()
        fire_lats = list()
        if len(fire_points_arr[latenc_asc_idxs[0]]) < 2:
            print("Only %d firepoints in lastest time csv." % len(fire_points_arr[latenc_asc_idxs[0]]))
            fire_points = fire_points_arr[latenc_asc_idxs[0]]
            for i in range(len(fire_points)):
                fire_lons.append(fire_points[i][0])
                fire_lats.append(fire_points[i][1])
        else:
            fire_points = fire_points_arr[latenc_asc_idxs[0]]
            merge_fireptrs_list = merge_pair_firepoints(fire_points, merge_rng, min_samples)    # 对最近时间的火点经纬度进行融合
            fire_lons = list()
            fire_lats = list()
            for fire_ptr in merge_fireptrs_list:
                fire_lons.append(fire_ptr[0])
                fire_lats.append(fire_ptr[1])
        if len(fire_lons) == 0:
            continue
        # 读取固定热源信息
        stable_heatptrs_xls = './fire_detection/suspicious_heatpoints.xlsx'
        fire_lons, fire_lats = filter_stable_heatptrs(stable_heatptrs_xls, fire_lons, fire_lats)
        # 读取地物信息
        rfid = open('./fire_detection/gm_lc_v3_1_2_parms.pkl', 'rb')
        data_dict = pkl.load(rfid)
        # 将火点判识结果导入数据库
        titles = ['日期', '时间', '经度', '纬度', '亚像元火点面积', '林地概率', '草地概率', '农田概率', '其他概率', '7波段像元值', '14波段像元值', '所在市', '所在县', '备注']
        fire_output_list = list()
        h = Himawari(pk_src_path + filename)
        for idx in range(len(fire_lons)):
            lat = fire_lats[idx]
            lon = fire_lons[idx]
            date = datetime.datetime.strptime(filename[4:12], "%Y%m%d")
            time = datetime.datetime.strptime(filename[12:16], "%H%M")
            sub_pxl_fire_area = 0
            comment = ""
            forestland, grassland, farmland, other = read_plant_info(lon, lat, data_dict)
            ch7_data = h.band_data_search(pk_path + filename[:16] + ".pk", 7, lon, lat)
            ch14_data = h.band_data_search(pk_path + filename[:16] + ".pk", 14, lon, lat)
            city, district = info_search(lon, lat)
            fire_output_list.append([date, time, lon, lat, sub_pxl_fire_area, forestland, grassland, farmland, other, ch7_data, ch14_data, city, district, comment])
        fire_output_df = pd.DataFrame(data=np.array(fire_output_list), columns=titles)
        tmp_thred = np.logical_and((fire_output_df['林地概率'] + fire_output_df['草地概率'] + fire_output_df['农田概率']) >= 30, fire_output_df['其他概率'] <= 10)
        fire_output_df = fire_output_df.loc[tmp_thred]
        database = Database(r"121.37.246.140", "FireDetection", "gdut", "fire#319")
        database.df_insert(fire_output_df, "fire_output")
        database.close()


if __name__ == '__main__':
    fire_timemerge()
