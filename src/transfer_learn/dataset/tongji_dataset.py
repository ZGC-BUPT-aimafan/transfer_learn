import json
import os


def count_json_elements(directory):
    # 查看每个json文件中有多少个元素
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 确保文件是一个 JSON 文件
        if filename.endswith(".json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 输出该 JSON 文件的元素个数
                    print(f"{filename}: {len(data)} elements")
            except Exception as e:
                print(f"Error reading {filename}: {e}")


if __name__ == "__main__":
    directory = "/mnt/20TB/aimafan/traffic_datasets/best_15_website/novpn/json"
    count_json_elements(directory)
