import os
import subprocess
from datetime import datetime
import time
import sys

def prep_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def main(args):
    start = time.time()
    if args[1] == "hf":
        MODELS = ["Deepseek_VL2", "Deepseek_Janus_1b", "Deepseek_Janus_7b", "Qwen2_VL", "Qwen2_5_VL_3b", "Qwen2_5_VL_7b", "LLaVA_One_Vision", "LLaVA_Next_Video"]
        PYTHONV = ["3.8"] * 3 + ["3.10"] * 5
    elif args[1] == "vllm":
        MODELS = ["Deepseek_VL2", "Qwen2_VL", "Qwen2_5_VL_3b", "Qwen2_5_VL_7b", "LLaVA_One_Vision"]
        PYTHONV = ["3.10"] * 5
    else:
        print("Specify a backend to test: 'hf' or 'vllm'")
        return 1
    INPUTPATH = "/home/allen.fang/big_bunny_1min_30fps.mp4"
    MASTER = "vllm"
    prep_dir(MASTER)

    BATCHSIZE = 1
    # keep going until no model runs properly
    one_pass = True
    while one_pass:
        one_pass = False
        dir_name = os.path.join(MASTER, f"BATCH_{BATCHSIZE}")
        # create if not created already
        prep_dir(dir_name)
        for i, model in enumerate(MODELS):
            print(f"Testing {model} on batch size {BATCHSIZE}, timestamp:", datetime.now().strftime("%H:%M:%S"))
            OUTPUTPATH = os.path.join(dir_name, MODELS[i])
            command = "python3" if PYTHONV[i] == "3.8" else "python3.10"
            try:
                result = subprocess.run([command, "tester.py",
                                INPUTPATH,
                                MODELS[i],
                                str(BATCHSIZE),
                                OUTPUTPATH],
                                check=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
                if result.returncode != 0:
                    raise RuntimeError(f"Subprocess failed with return code {result.returncode}. STDERR: {result.stderr}")
                one_pass = True
                print("Success, timestamp:", datetime.now().strftime("%H:%M:%S"))
            except subprocess.CalledProcessError as e:
                with open(OUTPUTPATH + ".log", "w") as file:
                    file.write(f"Command failed with return code {e.returncode}\n")
                    file.write(f"Error output:\n")
                    file.write("\n".join(e.stderr.decode().splitlines()) + "\n")
                print("Failed, timestamp:", datetime.now().strftime("%H:%M:%S"))
            except Exception as e:
                with open(OUTPUTPATH + ".log", "w") as file:
                    file.write(f"Unexpected error: {str(e)}\n")
                print("Failed with unexpected error, timestamp:", datetime.now().strftime("%H:%M:%S"))
        print("Finished batch", BATCHSIZE, "in", format_time(time.time() - start))
        BATCHSIZE += 1
    print("Finished in", format_time(time.time() - start))

if __name__ == "__main__":
    main(sys.argv)

