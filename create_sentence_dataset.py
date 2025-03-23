import csv
import os 

FOLDER_PATH = r"D:\diplomski\datasets\mathematics_dataset-v1.0"
SUBFOLDERS = ["train-easy", "train-medium", "train-hard"]
VALIDATION_SUBFOLDER = "interpolate"
CLASSES = {
    "algebra" : 0,
    "arithmetic" : 1, 
    "calculus" : 2, 
    "comparison" : 3, 
    "measurement" : 4,
    "numbers" : 5, 
    "polynomials" : 6, 
    "probability" : 7,
}

def create_sentences_csv():
    with open(os.path.join(FOLDER_PATH, 'sentences.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["Sentence", "label_text", "label_num"])
        for subfolder in SUBFOLDERS:
            files = os.listdir(os.path.join(FOLDER_PATH, subfolder))
            for file in files:
                print(f"Processing file {file} from subfolder {subfolder}...")
                with open(os.path.join(FOLDER_PATH, subfolder, file), 'r') as f:
                    #read every uneven line
                    for i, line in enumerate(f):
                        if i % 2 == 0:
                            writer.writerow([line.strip(), file.split("__")[0], CLASSES[file.split("__")[0]]])

def create_sentences_validation_csv():
    with open(os.path.join(FOLDER_PATH, 'sentences_validation.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["Sentence", "label_text", "label_num"])
        files = os.listdir(os.path.join(FOLDER_PATH, VALIDATION_SUBFOLDER))
        for file in files:
            print(f"Processing file {file} from subfolder {VALIDATION_SUBFOLDER}...")
            with open(os.path.join(FOLDER_PATH, VALIDATION_SUBFOLDER, file), 'r') as f:
                #read every uneven line
                for i, line in enumerate(f):
                    if i % 2 == 0:
                        writer.writerow([line.strip(), file.split("__")[0], CLASSES[file.split("__")[0]]])

if __name__ == "__main__":
    create_sentences_csv()
    create_sentences_validation_csv()