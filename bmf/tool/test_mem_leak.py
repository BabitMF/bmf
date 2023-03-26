import os
import json

def get_case():
    case_file_name = "./mem_leak_case.json"
    with open(case_file_name, mode='r') as f:
        config_dict = json.loads(f.read())
        if 'check_dir_list' in config_dict:
            check_dir_list = config_dict["check_dir_list"]
            return check_dir_list


dir_list = get_case()

log_prefix = "mem_leak_log"
if not os.path.isdir(log_prefix):
    os.mkdir(log_prefix)

for dir in dir_list:
    log_dir = log_prefix + "/" + dir["path"].split("/")[-1]
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    for case in dir["case"]:
        case_name = case.split(".")[-1]
        cmd = "./mem_leak.sh " + dir["path"] + " " + log_dir + " " + case + " " + case_name
        os.system(cmd)