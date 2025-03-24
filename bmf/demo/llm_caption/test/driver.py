import os
import subprocess

def prep_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)

def main():
    MODELS = ["Deepseek_VL2", "Deepseek_Janus_3b", "Deepseek_Janus_7b", "Qwen2_VL", "Qwen2_5_VL_3b", "Qwen2_5_VL_7b"]
    PYTHONV = ["3.8", "3.8", "3.8", "3.10", "3.10", "3.10"]
    INPUTPATH = "/home/allen.fang/big_bunny_1min_30fps.mp4"

    # keep going until no model runs properly
    one_pass = True
    while one_pass:
        one_pass = False
        BATCHSIZE = 1 
        dir_name = f"BATCH_{BATCHSIZE}"
        # create if not created already
        prep_dir(dir_name)
        for i, model in enumerate(MODELS):
            print(f"Testing {model} on batch size {BATCHSIZE}")
            OUTPUTPATH = os.path.join(dir_name, MODELS[i])
            command = "python3" if PYTHONV[i] == "3.8" else "python3.10"
            try:
                result = subprocess.run([command, "tester.py",
                                INPUTPATH,
                                MODELS[i],
                                str(BATCHSIZE),
                                OUTPUTPATH])
                if result.returncode != 0:
                    raise RuntimeError(f"Subprocess failed with return code {result.returncode}. STDERR: {result.stderr}")
                one_pass = True
                print("Success")
            except Exception as e:
                with open(OUTPUTPATH + ".log") as file:
                    file.write(str(e))
                break
                print("Failed")
        BATCHSIZE += 1

if __name__ == "__main__":
    main()

