import numpy as np
import firepoints_timemerge
import firepoints_detection
import time
import os
import argparse
from re import split

# 默认参数列表
default_dict = {
    "有效像元比例": 0.7,
    "云边判识云像元数量": 6,
    "绝对火点阈值": 330,
    "非火点阈值": 260,
    "7通道亮温增量": 20,
    "7-14通道亮温增量": 20,
    "火点融合时间": 1300,
    "火点融合经纬差值": 0.02
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", "--src_path", dest="pk_src_path", help="pk文件路径", default="./fire_detection/H8_pk/",
                        type=str)
    parser.add_argument("-dst", "--dst_path", dest="img_dst_path", help="图片输出路径", default="./Image/", type=str)
    parser.add_argument("-s", "--scheme", dest="scheme", help="火点判识方案", type=str)
    args = parser.parse_args()
    pk_src_path = args.pk_src_path
    img_dst_path = args.img_dst_path
    # 初始化可调参数列表
    if not args.scheme:
        param_dict = default_dict
    else:
        scheme = args.scheme
        # 格式化scheme变量，获得可调整参数列表
        scheme = split(",?:?", scheme[1:-1])
        param_dict = {scheme[i]: float(scheme[i + 1]) for i in range(0, len(scheme), 2)}

    if not os.path.exists(pk_src_path) or not os.path.exists(img_dst_path):
        print("ERROR:please enter a correct path")
        return

    while True:
        record_list = firepoints_detection.main(pk_src_path, img_dst_path, param_dict)   # 火点判识模块
        firepoints_timemerge.fire_timemerge(record_list, pk_src_path, param_dict)  # 多时相火点融合模块
        if record_list.size > 0:
            file_tmp = record_list[0]
            file_list = np.array(os.listdir(pk_src_path))
            idx = np.where(file_list == file_tmp)[0].item(0)
            if idx and idx - 6 > 0:
                for i in range(idx - 6):
                    os.remove(pk_src_path + file_list[i])

        time.sleep(10)


if __name__ == '__main__':
    main()
