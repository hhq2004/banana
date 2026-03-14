results_path = "/home/youzirui/code_dir_yzr/ori_img/IMG2XML_litserve/input/test01.pkl"

import pickle

with open(results_path, "rb") as f:
    results = pickle.load(f)

print(results)