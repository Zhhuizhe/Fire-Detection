import os
import pickle
import numpy as np
import cv2 as cv
from re import split
from osgeo import gdal
from osgeo import osr

c1 = 3.741832e-16
c2 = 1.438786e-2

"""
def read_dat():
    bands_data_list = list()
    file_bands_list = os.listdir("./Dat")
    for file in file_bands_list:
        band_info = np.zeros((1000, 1100), dtype=float)
        bz_file = bz2.BZ2File("./Dat/" + file)
        band_no = int(file[18:20]) - 1

        for i in range(1000):
            data = bz_file.read(2)
            y = int(i / 1100)
            x = int(i % 1100)
            if band_no <= 5:
                band_info[y][x] = int.from_bytes(data, byteorder='little', signed=True)
            else:
                band_info[y][x] = int.from_bytes(data, byteorder='little', signed=True)
                print(band_info[y][x])
        bands_data_list.append(band_info)
"""


class Himawari(object):
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            print("ERROR:未发现葵花8号卫星数据")
            return
        self.filename = os.path.split(filepath)[1].split(".")[0]
        h8_file = open(filepath, "rb")
        self.h8_data = pickle.load(h8_file)
        self.longitude_arr = self.h8_data[0]
        self.latitude_arr = self.h8_data[1]
        self.ch1_refl = self.h8_data[2]
        self.ch2_refl = self.h8_data[3]
        self.ch3_refl = self.h8_data[4]
        self.ch4_refl = self.h8_data[5]
        self.ch6_refl = self.h8_data[7]
        self.ch7_brt = self.h8_data[8]
        self.ch14_brt = self.h8_data[15]
        self.ch15_brt = self.h8_data[16]
        self.ch7_14_brt = self.ch7_brt - self.ch14_brt
        self.ch3_plus_4_refl = self.ch3_refl + self.ch4_refl
        self.width = len(self.longitude_arr)
        self.height = len(self.latitude_arr)
        self.__geoTrans = np.array([96, 0.01, 0, 30, 0, -0.01])
        h8_file.close()

    def create_pk(self):
        pass

    def __create_fire_img(self):
        lut = np.loadtxt("./FireImgLUT.txt", delimiter=",")
        ch7_brt = np.clip(self.ch7_brt * 10 + 0.5, 2200, 3400)
        ch3_refl = np.clip(self.ch3_refl * 1000 + 0.5, 0, 1000)
        ch4_refl = np.clip(self.ch4_refl * 1000 + 0.5, 0, 1000)
        # 最大最小值拉伸
        red = (255 * (ch7_brt - 2200) / (3400 - 2200)).astype(int)
        green = (255 * (ch4_refl - 0) / (1000 - 0)).astype(int)
        blue = (255 * (ch3_refl - 0) / (1000 - 0)).astype(int)
        # LUT
        img = np.zeros((self.height, self.width, 3))
        img[:, :, 0] = lut[2][blue]
        img[:, :, 1] = lut[1][green]
        img[:, :, 2] = lut[0][red]
        cv.imwrite("./Image/fire_img/" + self.filename + "_fire.png", img)
        return img

    def __create_true_color_img(self):
        ch1_refl = self.ch1_refl  # green
        ch2_refl = self.ch2_refl  # blue
        ch3_refl = self.ch3_refl  # red
        # 最大最小值拉伸
        minimn = np.min([np.min(ch1_refl), np.min(ch2_refl), np.min(ch3_refl)])
        maximn = np.max([np.max(ch1_refl), np.max(ch2_refl), np.max(ch3_refl)])
        blue = (ch1_refl - minimn) / (maximn - minimn)
        green = (ch2_refl - minimn) / (maximn - minimn)
        red = (ch3_refl - minimn) / (maximn - minimn)
        img = np.zeros((self.height, self.width, 3))
        img[:, :, 0] = np.clip(np.power(blue, 0.67) * 1.5, 0, 1)
        img[:, :, 1] = np.clip(np.power(green, 0.67) * 1.5, 0, 1)
        img[:, :, 2] = np.clip(np.power(red, 0.67) * 1.5, 0, 1)
        return img * 255

    def __create_ch_img(self):
        ch3_refl = self.ch3_refl
        ch7_brt = self.ch7_brt
        ch7_14_brt = self.ch7_14_brt
        # 获取3通道图像
        ch3_refl = (ch3_refl - np.min(ch3_refl)) / (np.max(ch3_refl) - np.min(ch3_refl))
        ch3_refl = np.clip(ch3_refl, 0, 1) * 255
        # 获取7通道图像
        ch7_brt = 255 * (ch7_brt - np.min(ch7_brt)) / (np.max(ch7_brt) - np.min(ch7_brt))
        # 获取7-14通道图像
        ch7_14_brt = 255 * (ch7_14_brt - np.min(ch7_14_brt)) / (np.max(ch7_14_brt) - np.min(ch7_14_brt))
        return ch3_refl, ch7_brt, ch7_14_brt

    def __create_16_ch_img(self, dst_path):
        file_format = "GTiff"
        driver = gdal.GetDriverByName(file_format)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dst_ds = driver.Create(dst_path, xsize=self.width, ysize=self.height, bands=16, eType=gdal.GDT_Int16)
        dst_ds.SetGeoTransform(self.__geoTrans)
        dst_ds.SetProjection(srs.ExportToWkt())

        for i in range(1, 17):
            h8_data_temp = self.h8_data[i + 1]
            if i <= 6:
                h8_data_temp = h8_data_temp * 1000 + 0.5
            else:
                h8_data_temp = h8_data_temp * 10 + 0.5
            dst_ds.GetRasterBand(i).WriteArray(h8_data_temp)

    def create_img(self, dst_path="./Image/"):
        if not os.path.exists(dst_path):
            print("ERROR:please input a correct image path")
            return
        path_list = [dst_path + "true_color_img/",
                     dst_path + "fire_img/",
                     dst_path + "cloud_img/",
                     dst_path + "ch7/",
                     dst_path + "ch7_14/",
                     dst_path + "16ch_tiff/"]
        for path in path_list:
            if not os.path.exists(path):
                os.mkdir(path)
        cloud_img, ch7_img, ch7_14_img = self.__create_ch_img()
        true_color_img = self.__create_true_color_img()
        fire_img = self.__create_fire_img()
        path_list = [dst_path + "true_color_img/" + self.filename + "_true_color.png",
                     dst_path + "fire_img/" + self.filename + "_fire.png",
                     dst_path + "cloud_img/" + self.filename + "_cloud.png",
                     dst_path + "ch7/" + self.filename + "_ch7.png",
                     dst_path + "ch7_14/" + self.filename + "_ch7_14.png",
                     dst_path + "16ch_tiff/" + self.filename + "_16_ch.tiff"]

        cv.imwrite(path_list[0], true_color_img)
        cv.imwrite(path_list[1], fire_img)
        cv.imwrite(path_list[2], cloud_img)
        cv.imwrite(path_list[3], ch7_img)
        cv.imwrite(path_list[4], ch7_14_img)
        # 创建16通道TIFF
        self.__create_16_ch_img(path_list[5])
        return path_list

    # 计算亚像元火点
    def sub_pxl_fire_field(self, ch4_bt, ch7_bt, ch4_w, ch7_w, background_bt):

        def planck_func(w, t):
            return c1 * np.power(w, -5) / (np.exp(c2 / (w * t)) - 1)

        # 初始化算法参数
        idx = 0
        iterations = 100
        delta = np.ones(2)
        x_init = np.array([0.01, 300])
        while idx < iterations and np.sum(delta) > 1e-15:
            p = x_init[0]
            target_bt = x_init[1]
            rad = np.array([planck_func(ch4_w, ch4_bt), planck_func(ch7_w, ch7_bt)])
            rad_bg = np.array([planck_func(ch4_w, background_bt), planck_func(ch7_w, background_bt)])
            rad_tar = np.array([planck_func(ch4_w, target_bt), planck_func(ch7_w, target_bt)])
            f = p * rad_tar + (1 - p) * rad_bg - rad
            tmp = np.array([rad_tar[0] * (- c2 * np.exp(c2 / (ch4_w * target_bt)) / (ch4_w * np.power(target_bt, 2))),
                            rad_tar[0] * (- c2 * np.exp(c2 / (ch7_w * target_bt)) / (ch7_w * np.power(target_bt, 2)))])
            inverse_jacobian = np.linalg.inv(np.vstack((rad_tar - rad_bg, tmp)).T)
            print(inverse_jacobian)
            delta = - np.dot(inverse_jacobian, f)
            x_init = x_init + delta
            idx += 1
        return x_init

    # 根据经纬坐标查值
    def band_data_search(self, band, longitude, latitude):
        if band <= 0 or band > 16:
            print("ERROR: invalid band no")
            return
        h8_band_data = self.h8_data[band + 1]
        ret = h8_band_data[int((30 - latitude) / 0.01)][int((longitude - 96) / 0.01)]
        return ret


if __name__ == '__main__':
    pass
