# 分两步，第一步处理为json文件，第二步处理为npy文件

import json
import os

import numpy as np
from pypcaptools import PcapHandler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from wechat_fang import set_wechat

from transfer_learn.myutils import project_path

# from transfer_learn.myutils.logger import logger


def merge_json(input_files, dir_name):
    merged_data = []

    # 读取每个文件并合并数据
    for file_name in input_files:
        with open(file_name, "r") as file:
            datas = json.load(file)
            for data in datas:
                merged_data.append(data)  # 将每个文件的数据添加到列表中
        os.remove(file_name)

    # 将合并后的数据写入新的JSON文件
    with open(dir_name, "w") as outfile:
        json.dump(merged_data, outfile, indent=4)  # indent参数用于美化输出


def pcap2json_split_session(origin_root, output_root):
    class_dics = {}
    for root, _, files in os.walk(origin_root):
        for file_name in files:
            if file_name.endswith(".pcap"):
                if root.split("/")[-1] not in class_dics:
                    class_dics[root.split("/")[-1]] = []
                pcap_path = os.path.join(root, file_name)
                ph = PcapHandler(pcap_path)
                _, output_path = ph.split_flow(output_root, output_type="json")
                class_dics[root.split("/")[-1]].append(output_path)

    for key in class_dics:
        output_path = os.path.join(output_root, key + ".json")
        merge_json(class_dics[key], output_path)


def json2npy_df_payload_not_zero(origin_root, output_root, lens: list, min_len, type):
    # 每一个类别一个json文件，一个json文件中包括若干流，类别名就是label。
    # 提取特征，按照train:test:val = 8:1:1进行分割，保存为npy文件，npy文件中是字典，key分别是"X"和"y"
    # 取payload长度序列，去掉0
    # lens是一个整型列表，表示取前多少个包
    for length in tqdm(lens, total=len(lens)):
        result = {"X": [], "y": []}
        label = -1
        for root, _, files in os.walk(origin_root):
            for file_name in files:
                if file_name.endswith(".json"):
                    # 获得文件名，不带后缀
                    label += 1

                    with open(os.path.join(output_root_vpn, file_name), "r") as file:
                        datas = json.load(file)
                        for data in datas:
                            payload_len = data.get("payload", [])
                            # 过滤0
                            filtered_payload_len = [
                                int(value)
                                for value in payload_len
                                if value != "0" and value != "+0" and value != "-0"
                            ]
                            # 调整长度
                            if len(filtered_payload_len) < min_len:
                                continue
                            if len(filtered_payload_len) < length:
                                filtered_payload_len.extend(
                                    [0] * (length - len(filtered_payload_len))
                                )
                            else:
                                filtered_payload_len = filtered_payload_len[:length]
                            result["X"].append(filtered_payload_len)
                            result["y"].append(label)
        X = np.array(result["X"])
        y = np.array(result["y"])

        # 按8:1:1比例分割数据
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # 保存为npy文件
        np.save(
            os.path.join(output_root, f"train_{type}_{length}_payload.npy"),
            {"X": X_train, "y": y_train},
        )
        np.save(
            os.path.join(output_root, f"test_{type}_{length}_payload.npy"),
            {"X": X_test, "y": y_test},
        )
        np.save(
            os.path.join(output_root, f"val_{type}_{length}_payload.npy"),
            {"X": X_val, "y": y_val},
        )


if __name__ == "__main__":
    origin_root_vpn = "/mnt/20TB/aimafan/traffic_datasets/best_15_website/vpn/pcap"
    origin_root_novpn = "/mnt/20TB/aimafan/traffic_datasets/best_15_website/novpn/pcap"
    output_root_vpn = "/mnt/20TB/aimafan/traffic_datasets/best_15_website/vpn/json"
    output_root_novpn = "/mnt/20TB/aimafan/traffic_datasets/best_15_website/novpn/json"

    json2npy_df_payload_not_zero(
        output_root_vpn,
        os.path.join(project_path, "data", "dataset"),
        [100, 200, 300, 400, 500],
        "vpn",
    )
    set_wechat("VPN数据搞完了")
    json2npy_df_payload_not_zero(
        output_root_novpn,
        os.path.join(project_path, "data", "dataset"),
        [100, 200, 300, 400, 500],
        "novpn",
    )
    set_wechat("non-VPN数据搞完了")
