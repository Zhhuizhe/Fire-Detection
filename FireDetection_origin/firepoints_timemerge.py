# -*- coding: utf-8 -*- 
import os
import os.path as osp

import sys
import glob

import time
import datetime
import argparse

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
import pickle as pkl
from scipy.spatial import distance as dist


def get_merge_csvdata(firetimes, firecsvpaths, out_csv_datetime, mergetime):
    merge_stop_time = out_csv_datetime
    merge_start_time = out_csv_datetime - datetime.timedelta(seconds=mergetime)

    within_mergetime_bools = [fire_time >= merge_start_time and fire_time <= merge_stop_time \
                              for fire_time in firetimes]
    within_mergetime_csvpaths = [firecsvpaths[idx] for idx in range(len(firecsvpaths)) \
                                 if within_mergetime_bools[idx]]
    within_mergetime_datetimes = [firetimes[idx] for idx in range(len(firecsvpaths)) \
                                  if within_mergetime_bools[idx]]

    if len(within_mergetime_csvpaths) <= 0:
        return np.array([]), np.array([]), None
    
    fire_points_arr_ = list()

    for file_path, fire_time in zip(within_mergetime_csvpaths, within_mergetime_datetimes):
        file_path = file_path.strip()
        print("processing csv: %s" % file_path)

        cur_encoding = 'GB2312'
        csv_fdata = pd.read_csv(file_path, encoding=cur_encoding)
        cur_csv_lons = csv_fdata[r'Lons'].values.flatten()
        cur_csv_lats = csv_fdata[r'Lats'].values.flatten()
#        cur_csv_probs = csv_fdata[r'平均可信度(%)'].values.flatten()

        fire_points = list()
        for lon, lat in zip(cur_csv_lons, cur_csv_lats):
            fire_points.append([lon, lat])

        fire_points_arr_.append(np.array(fire_points))

    fire_times_ = np.array(within_mergetime_datetimes)
    fire_points_arr_ = np.array(fire_points_arr_)

    indices = np.argsort(fire_times_)
    max_time = fire_times_[indices[-1]]
    fire_times_ = (max_time - fire_times_)

    fire_times = np.array([fire_times_[idx].total_seconds() * -1 for idx in indices])
    fire_points_arr = fire_points_arr_[indices]

    fire_times_list = list(); fire_points_list = list()
    for fire_time, fire_points in zip(fire_times, fire_points_arr):
        if np.abs(fire_time) <= mergetime:
            fire_times_list.append(fire_time)
            fire_points_list.append(fire_points)

    return np.array(fire_times_list), np.array(fire_points_list), within_mergetime_csvpaths[indices[-1]]


#def merge_pair_firepoints(fire_points, merge_rng=0.02):
#    if len(fire_points) <= 0:
#        return pivot_points
#    else: 
#        fire_points_array = np.vstack([item for item in fire_points if len(item) > 0])
#        db_obj = DBSCAN(eps=merge_rng, min_samples=1).fit(fire_points_array)
#        clust_ptrlbls = db_obj.labels_
#
#        # preserve the clusters with more than 1 points
#        uniq_clust_ptrlbls = np.unique(clust_ptrlbls)
#        merge_point_idxs = list()
#
#        for clust_lbl in uniq_clust_ptrlbls:
#            cur_clust_idxs = np.where(clust_ptrlbls == clust_lbl)[0]
#            cur_clust_nelms = len(cur_clust_idxs)
#
#            if cur_clust_nelms > 1:
#                merge_point_idxs.append(cur_clust_idxs[0])
#
#        merge_points = [(fire_points_array[idx, 0], fire_points_array[idx, 1]) 
#                        for idx in merge_point_idxs]
#        merge_points = np.array(merge_points)
#
#        return merge_points

def merge_pair_firepoints(fire_points, merge_rng, min_samples):
    if len(fire_points) <= 0:
        return pivot_points
    else: 
        fire_points_array = np.vstack([item for item in fire_points if len(item) > 0])
        db_obj = DBSCAN(eps=merge_rng, min_samples=min_samples).fit(fire_points_array)
        clust_ptrlbls = db_obj.labels_

        # preserve the clusters with more than 1 points
        uniq_clust_ptrlbls = np.unique(clust_ptrlbls)
        merge_point_idxs = list()

        for clust_lbl in uniq_clust_ptrlbls:
            cur_clust_idxs = np.where(clust_ptrlbls == clust_lbl)[0]
            cur_clust_nelms = len(cur_clust_idxs)

            if cur_clust_nelms > 1:                      #单点都不要了
                if (fire_points_array[cur_clust_idxs[0],0] - fire_points_array[cur_clust_idxs[-1],0])> 0.023 or (fire_points_array[cur_clust_idxs[0],1] - fire_points_array[cur_clust_idxs[-1],1]>0.023):
#                    merge_point_idxs.append(cur_clust_idxs[0])
                    merge_point_idxs.append(cur_clust_idxs[-1])
                else:
                    merge_point_idxs.append(cur_clust_idxs[-1])
#                    merge_point_idxs.append(cur_clust_idxs[0])


        merge_points = [(fire_points_array[idx, 0], fire_points_array[idx, 1]) 
                        for idx in merge_point_idxs]
        merge_points = np.array(merge_points)

        return merge_points


#读取植被信息
def ReadClass(Xgeo, Ygeo, data_dict):
    row = data_dict['row']; col = data_dict['col']
    GT = data_dict['GT']; data = data_dict['data']

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
            if data[irow-1+i,icol-1+j] <6:
                forestland += 1
            elif  data[irow-1+i,icol-1+j] >5 and data[irow-1+i,icol-1+j] <11:
                grassland += 1
            elif data[irow-1+i,icol-1+j] >10 and data[irow-1+i,icol-1+j] <16:
                farmland += 1
            else:
                other += 1

    return int(forestland/0.16),int(grassland/0.16),int(farmland/0.16),int(other/0.16)


def save_fire_txt(Rs, txt_path, plant_data, probs, b_plant_filt=True):
    strrect = '序号,中心经度,中心纬度,热点像元个数,热点面积（公顷）,林地概率(%),草地概率(%),农田概率(%),其他概率(%),平均可信度(%),备注\n'
    fire_lons = Rs[0]; fire_lats = Rs[1]
    
    if not (len(fire_lons) <= 0 or len(fire_lats) <= 0):
        for i_iter in range(len(fire_lons)):
            lon = fire_lons[i_iter]; lat = fire_lats[i_iter]
            forestland,grassland,farmland,other = ReadClass(lon, lat, plant_data)

            if np.sum([forestland, grassland, farmland]) < 30 and other > 10 and b_plant_filt:
                pass

            strrect += str(i_iter+1) + ',' + '%.3f' % (lon) + ',' + '%.3f' % (lat) + ',0,0,'+str(forestland)+','+str(grassland)+','+str(farmland)+','+str(other)+',' + str(probs[i_iter]) + ', '

            if (i_iter != len(fire_lons)-1):
                strrect += '\n'

    with open(txt_path,"w") as f:
        f.write(strrect)


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


if __name__ == '__main__':
#     Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict fire position.')
    parser.add_argument('--mergedir', help='previous fire result csv time')
    parser.add_argument('--mergetime', help='previous fire result csv time')
    parser.add_argument('--min_samples', help='min_samples for dbscan')
##    parser.add_argument('--stablehp', help='table file recording stable heat points')
##    parser.add_argument('--pfilt', help='filtering fire points using plant category distribution')
##    parser.add_argument('outcsv_path', help='file name of the merged csv path')
#    args = parser.parse_args()
#
    mergedir = args.mergedir
    mergetime = eval(args.mergetime)
    min_samples = eval(args.min_samples)
#    mergetime = 1300        # 融合
#    merge_rng = 0.02
#    min_samples = 1
#    stable_heatptrs_xls = args.stablehp
    b_plant_filt = True
#    outcsv_path = args.outcsv_path

    mergedir = './fire_detection/output_eliminate/'

    if not os.path.exists(mergedir):
        sys.exit(0)
    firstStageCsv_path = './fire_detection/output_eliminate/'
  
    
    firstStageCsv_list2 = os.listdir(firstStageCsv_path)
    filelist_dbscan = os.listdir('./fire_detection/finalOutput_dbscan/')
    for fileTemp in firstStageCsv_list2:
        if fileTemp not in filelist_dbscan:
            output_csv_datetime = datetime.datetime.strptime(fileTemp[4:16], '%Y%m%d%H%M')
            
#        firstStageCsv_list = firstStageCsv_list2[1:i+2]
#        timeList = []
#        for firstStageCsv in firstStageCsv_list:
#            timeList.append(datetime.datetime.strptime(firstStageCsv[4:16], '%Y%m%d%H%M'))
        output_csv_datetime = max(timeList)

    # collect the pre-merge csv files and data by the output csv datetime information
        all_fire_csvs = glob.glob(osp.join(mergedir, '*.csv'))
        all_fire_times = list()
    
        for file_path in all_fire_csvs:
            file_path = file_path.strip()
            print("processing csv: %s" % file_path)
    
            if os.path.isfile(file_path) == False:
                continue
            #import pdb;pdb.set_trace()
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            file_name = file_name.replace('SHJC', '')
            file_name = file_name[:len(file_name)-1]
    
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
            sys.exit(-1)
    
        fire_times, fire_points_arr, last_fname = get_merge_csvdata(all_fire_times, all_fire_csvs, output_csv_datetime, mergetime)
    
        if len(fire_times) <= 0:
            print("No files in the source directory")
            sys.exit(-1)
        elif len(fire_times) <= 1:
            print("No enough files in the source directory")
            sys.exit(-1)
    
        latenc_asc_idxs = np.argsort(fire_times)[::-1]
        if len(fire_points_arr[latenc_asc_idxs[0]]) < 2:
            print("Only %d firepoints in lastest time csv." % len(fire_points_arr[latenc_asc_idxs[0]]))
    
            fire_lons = list(); fire_lats = list(); probs = list()
            fire_points = fire_points_arr[latenc_asc_idxs[0]]
    
            for i in range(len(fire_points)):
                fire_lons.append(fire_points[i][0])
                fire_lats.append(fire_points[i][1])
                probs.append(np.random.randint(60, 90))
    
        else:
            fire_points = fire_points_arr[latenc_asc_idxs[0]]
            merge_fireptrs_list = merge_pair_firepoints(fire_points, merge_rng, min_samples)
    
            fire_lons = list(); fire_lats = list(); probs = list()
            for fire_ptr in merge_fireptrs_list:
                fire_lons.append(fire_ptr[0])
                fire_lats.append(fire_ptr[1])
                probs.append(np.random.randint(60, 90))
    
        with open('./fire_detection/gm_lc_v3_1_2_parms.pkl', 'rb') as rfid:
            data_dict = pkl.load(rfid)
    
        b_filter_stablehps = True
        if b_filter_stablehps:
            stable_heatptrs_xls = './fire_detection/suspicious_heatpoints.xlsx'
            fire_lons, fire_lats = filter_stable_heatptrs(stable_heatptrs_xls, fire_lons, fire_lats)
    
        csv_fire_path = './fire_detection/finalOutput_dbscan/'+fileTemp
        save_fire_txt([fire_lons, fire_lats], csv_fire_path, data_dict, probs, b_plant_filt=True)