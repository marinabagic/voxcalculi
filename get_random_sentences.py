import os
import random
import glob

directory = 'D:\\diplomski\\datasets\\mathematics_dataset-v1.0\\train-easy'

txt_files = glob.glob(os.path.join(directory, '*.txt'))

random_rows = []

for txt_file in txt_files:
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    training_data = lines[::2]

    random_row = random.choice(training_data)
    random_rows.append(random_row)

with open('random_rows.txt', 'w') as file:
    for i, row in enumerate(random_rows):
        module = txt_files[i].split(os.sep)[-1].split('.')[0]
        file.write(f"{module} & {row}")
