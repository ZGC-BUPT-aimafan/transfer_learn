import os

import numpy as np
from wechat_fang import set_wechat

from transfer_learn.df.df_nodev_train import train, verification
from transfer_learn.myutils import project_path
from transfer_learn.myutils.logger import logger

if __name__ == "__main__":
    root_handled_path = "/home/aimafan/Documents/mycode/transfer_learn/data/dataset"
    LENGTHs = ["100", "200", "300", "400", "500"]
    for LENGTH in LENGTHs:
        TYPE = "novpn"
        MODE = "payload"

        merge = TYPE + "_" + LENGTH + "_" + MODE
        train_path = os.path.join(root_handled_path, "train" + "_" + merge + ".npy")
        test_path = os.path.join(root_handled_path, "test" + "_" + merge + ".npy")
        val_path = os.path.join(root_handled_path, "val" + "_" + merge + ".npy")

        train_data = np.load(train_path, allow_pickle=True).item()
        X_train = train_data["X"]
        y_train = train_data["y"]

        test_data = np.load(test_path, allow_pickle=True).item()
        X_test = test_data["X"]
        y_test = test_data["y"]

        val_data = np.load(val_path, allow_pickle=True).item()
        X_val = val_data["X"]
        y_val = val_data["y"]

        model_dir = os.path.join(project_path, "data", "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, merge + ".pth")
        logger.info(f"开始训练{merge}模型")
        accuracy = train(X_train, X_test, X_val, y_train, y_test, y_val, model_path, 36)

        logger.info(f"开始在vpn数据集上使用{merge}.pth模型进行测试")
        TYPE = "vpn"
        merge_vpn = TYPE + "_" + LENGTH + "_" + MODE
        vpn_path = os.path.join(root_handled_path, "train" + "_" + merge_vpn + ".npy")
        vpn_data = np.load(vpn_path, allow_pickle=True).item()
        X_vpn = vpn_data["X"]
        y_vpn = vpn_data["y"]
        print(vpn_path)
        valid_accuracy = verification(model_path, X_vpn, y_vpn)

        set_wechat(
            f"{merge}的df模型训练结束, 在No-VPN数据集上的准确率为{accuracy:.4f}，在VPN数据集上的准确率为{valid_accuracy:.4f}"
        )
        logger.info(
            f"{merge}的df模型训练结束, 在No-VPN数据集上的准确率为{accuracy:.4f}，在VPN数据集上的准确率为{valid_accuracy:.4f}"
        )
