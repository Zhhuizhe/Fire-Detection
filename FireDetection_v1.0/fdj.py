import numpy as np
import sys

if './' not in sys.path:
    sys.path.insert(0, './')


def get_kernal(size, band, x_start, y_start):
    kernal = band[x_start:x_start+size, y_start:y_start+size]         # 从起始点获得15*15的矩形区域
    return kernal


def calcu_band_features(band_no1, band_no2, band_no3, x_start, y_start, size, kernal_point):
    kernal_no1 = get_kernal(size, band_no1, x_start, y_start)    # 15*15的矩阵，矩形局部区域
    kernal_no2 = get_kernal(size, band_no2, x_start, y_start)
    kernal_no3 = get_kernal(size, band_no3, x_start, y_start)

    T_no1 = kernal_no1[kernal_point, kernal_point]         # 矩形局部区域中心点的值
    T_no2 = kernal_no2[kernal_point, kernal_point]
    T_no3 = kernal_no3[kernal_point, kernal_point]

    deta = T_no2 - T_no3                             # 中心点的差
    deta_kernal = kernal_no2 - kernal_no3            # 矩形局部区域的差

    return kernal_no2, deta_kernal, T_no2, deta, T_no1, T_no3


def nightBool_func(h8_band3_data, h8_band4_data, fire_lat_index, fire_lon_index):
    return abs(h8_band3_data[fire_lat_index, fire_lon_index]) < 0.01 and abs(h8_band4_data[fire_lat_index, fire_lon_index]) < 0.01


def nonCloudBool_func(h8_band3_data, h8_band4_data, h8_band15_data, fire_lat_index, fire_lon_index):
    nightBool = nightBool_func(h8_band3_data, h8_band4_data, fire_lat_index, fire_lon_index)
    nonCloudBool = (h8_band3_data[fire_lat_index, fire_lon_index] + h8_band4_data[fire_lat_index, fire_lon_index] < 1.2) \
    and (h8_band15_data[fire_lat_index, fire_lon_index] > 265) \
    and ((h8_band3_data[fire_lat_index, fire_lon_index]+h8_band4_data[fire_lat_index, fire_lon_index] < 0.7) or (h8_band15_data[fire_lat_index, fire_lon_index] > 285)) \
    or nightBool
    return nonCloudBool


def nonWaterBool_func(h8_band3_data,h8_band4_data,h8_band15_data,h8_band6_data,fire_lat_index,fire_lon_index):
    w1 = 0.05   # 白昼水阈值
    w2 = 0.01
    nonWaterBool = h8_band6_data[fire_lat_index, fire_lon_index] > w1 or nightBool_func(h8_band3_data,h8_band4_data,fire_lat_index,fire_lon_index)
    return nonWaterBool


# band_no1 - 通道3反射率, band_no2 - 通道7亮温值, band_no3 - 通道14亮温值, fire_lat_index - 当前火点纬度索引
# fire_lon_index - 当前火点经度索引, h8_band15_data - 通道15亮温值, h8_band34_data - 通道3与通道4的反射率之和
def hd_test(band_no1, band_no2, band_no3, fire_lat_index, fire_lon_index, h8_band4_data, h8_band15_data, h8_band6_data,h8_band34_data,h8_band2_data,kernel_cloud,size=15):
    kernal_point = int((size-1)/2)     # 默认为7
    x_start = fire_lat_index - kernal_point
    y_start = fire_lon_index - kernal_point
    # kernal_no2 - 通道7亮温窗口; deta_kernal - 通道7减通道14亮温差值窗口; deta - 通道7减通道14亮温差值
    # T_no2 - 通道7亮温值; T_no1 - 通道3反射率; T_no3 - 通道14亮温值
    kernal_no2, deta_kernal, T_no2, deta, T_no1, T_no3 = calcu_band_features(band_no1, band_no2, band_no3, x_start, y_start, size, kernal_point)
    # 通过通道3和通道4精细判云
    band_34 = h8_band34_data
    kernal_band34 = get_kernal(size, band_34, x_start, y_start)     # 获取通道3+通道4亮温窗口
    T_band34 = kernal_band34[kernal_point, kernal_point]
    n_cloud = (T_band34-kernal_band34.mean())/kernal_band34.std()
    
    band_2 = h8_band2_data
    kernal_band2 = get_kernal(size, band_2, x_start, y_start)
    T_band2 = kernal_band2[kernal_point, kernal_point]
    n_cloud_band2 = (T_band2-kernal_band2.mean())/kernal_band2.std()
    # 将窗口内非中心点保存至no_central列表中，将通道7符合阈值条件的点保存至valid列表中
    kernal_no2_valid = []
    deta_kernal_valid = []
    kernal_no2_no_central = []
    deta_kernal_no_central = []
    for i in range(kernal_no2.shape[0]):
        for j in range(kernal_no2.shape[1]):
            if not (i == int((kernal_no2.shape[0]-1)/2) and j == int((kernal_no2.shape[0]-1)/2)):                # 确保窗口中心点不被加入列表中
                kernal_no2_no_central.append(kernal_no2[i, j])
                deta_kernal_no_central.append(deta_kernal[i, j])
                if not (kernal_no2[i, j] > 325 and deta_kernal[i, j] > 20):
                    if kernel_cloud[i, j] == 0:
                        kernal_no2_valid.append(kernal_no2[i, j])
                        deta_kernal_valid.append(deta_kernal[i, j])

    kernal_no2_valid = np.array(kernal_no2_valid)
    deta_kernal_valid = np.array(deta_kernal_valid)
    n1_std_with_valid_pixels = (T_no2-kernal_no2_valid.mean())/kernal_no2_valid.std()
    n2_std_with_valid_pixels = (deta-deta_kernal_valid.mean())/deta_kernal_valid.std()
    # 夜判 0 - 白天  1 - 夜间
    # band_no1 - 通道3反射率; h8_band4_data - 通道4亮温值; fire_lat_index - 火点纬度索引值; fire_lon_index - 火点精度索引值
    nightBool = nightBool_func(band_no1, h8_band4_data, fire_lat_index, fire_lon_index)
    # 云判 1 - 非云  0 - 云
    nonCloudBool = nonCloudBool_func(band_no1, h8_band4_data, h8_band15_data, fire_lat_index, fire_lon_index)
    cloudBool = not nonCloudBool
    # 水判 1 - 非水  0 - 水
    nonWaterBool = nonWaterBool_func(band_no1, h8_band4_data, h8_band15_data, h8_band6_data, fire_lat_index, fire_lon_index)
    waterBool = not nonWaterBool
    # 云边判断
    size_size7 = 7
    kernal_point_size7 = int((size_size7-1)/2)     # 默认为7
    x_start_size7 = fire_lat_index - kernal_point_size7
    y_start_size7 = fire_lon_index - kernal_point_size7
    kernal_no2_size7, deta_kernal_size7, T_no2_size7, deta_size7, T_no1_size7, T_no3_size7 = \
                calcu_band_features(band_no1, band_no2, band_no3, x_start_size7, y_start_size7, size_size7, kernal_point_size7)
    
    return T_no1, T_no2, T_no3, n1_std_with_valid_pixels,n2_std_with_valid_pixels,deta_kernal.mean(),nightBool,cloudBool,waterBool,kernal_no2_size7.mean(),deta_kernal_size7.mean(),n_cloud,n_cloud_band2
