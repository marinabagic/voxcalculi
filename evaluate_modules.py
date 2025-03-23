import subprocess
import os

def get_last_accuracy(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()

        i = len(lines) - 1
        while i >= 0:
            last_line = lines[i].strip()
            if last_line.startswith("eval"):
                start_index = last_line.find("acc=")
                if start_index != -1:
                    end_index = last_line.find(' ', start_index)
                    if end_index == -1:
                        end_index = len(last_line)
                    try:
                        accuracy = float(last_line[start_index + 4:end_index])
                        return accuracy
                    except ValueError:
                        return None
            i -= 1
        
        return None

MODEL_PATHS = ["C:\\Users\\anteg\\Desktop\\PROJEKTI\\TP-Transformer-master\\trained\\algebra2d\\lr=0.0001_bs=128_h=512_f=2048_nl=6_nh=8_d=0.0_Adam_\\195932581\\best_eval_model.pt"]

for m_path in MODEL_PATHS:
    with open("C:\\Users\\anteg\\Desktop\\PROJEKTI\\TP-Transformer-master\\logs\\eval.txt", "a") as f:
        f.write(m_path + "\n")
    MODEL_PATH = m_path
    INFO = MODEL_PATH.split(os.sep)[-3]
    BATCH_SIZE = 64
    LR = INFO.split("_")[0].split("=")[1]
    HIDDEN_SIZE = INFO.split("_")[2].split("=")[1]
    FF_SIZE = INFO.split("_")[3].split("=")[1]
    N_LAYERS = INFO.split("_")[4].split("=")[1]
    N_HEADS = INFO.split("_")[5].split("=")[1]
    DROPOUT = INFO.split("_")[6].split("=")[1]
    print("LR: ", LR)
    print("HIDDEN_SIZE: ", HIDDEN_SIZE)
    print("FF_SIZE: ", FF_SIZE)
    print("N_LAYERS: ", N_LAYERS)
    print("N_HEADS: ", N_HEADS)
    print("DROPOUT: ", DROPOUT)

    modules = [name for name in os.listdir(r"D:\diplomski\datasets\dm_math") if os.path.isdir(os.path.join(r"D:\diplomski\datasets\dm_math", name))]
    for module in modules:
        if module.endswith("big") or module.endswith("longer") or module.endswith("more") or module.endswith("more_samples") or module=="measurement__conversion":
            model_name = "tp-transformer"
            COMMAND = f"python main.py --model_name={model_name} --module_name={module} --n_layers={N_LAYERS} --n_heads={N_HEADS} --hidden={HIDDEN_SIZE} --filter={FF_SIZE} --load_model={MODEL_PATH} --batch_size={BATCH_SIZE} --eval_mode".split(" ")
            subprocess.check_call(COMMAND, stdout=None, stderr=None)

            model_params = MODEL_PATH.split(os.sep)[-3].replace("bs=128", "bs=64")[:-1]
            log_file = os.path.join("C:\\Users\\anteg\\Desktop\\PROJEKTI\\TP-Transformer-master\\logs", module, model_name, model_params, MODEL_PATH.split(os.sep)[-2], "output.log")
            accuracy = get_last_accuracy(log_file)
            if accuracy is not None:
                with open("C:\\Users\\anteg\\Desktop\\PROJEKTI\\TP-Transformer-master\\logs\\eval.txt", "a") as f:
                    f.write(f"{module} accuracy: {accuracy}\n")
            else:
                print("No accuracy found in the last line.")




