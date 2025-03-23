import os
import torch

DATA_SOURCE_DIR = "D:\\diplomski\\datasets\\mathematics_dataset-v1.0"

DATASET_TARGET_DIR = "D:\\diplomski\\datasets\\dm_math"

TRAIN_SUB_DIRS = ["train-easy", "train-medium", "train-hard"]
INTER_SUB_DIRS = ["interpolate"]
EXTRA_SUB_DIRS = ["extrapolate"]

def read_files(subdirs, module_file):
  all_lines = []
  for subdir in subdirs:
    with open(os.path.join(DATA_SOURCE_DIR, subdir, module_file), "r") as f:
      lines = f.readlines()
    print("... read {} lines from {}".format(len(lines), subdir))
    all_lines += lines
  return all_lines

def split_into_x_y(lines):
  x, y = [], []
  for idx in range(0,len(lines),2):
    x.append(lines[idx])
    y.append(lines[idx+1])
  return x, y

def make_jit_pairs_and_indexes(x, y):
    xy_list = []
    indexes = []
    file_offset = 0
    vocab = set()

    for xx, yy in zip(x, y):
        x_line = xx.replace("\n", "")
        y_line = yy.replace("\n", "")

        vocab.update(list(x_line))
        vocab.update(list(y_line))

        assert not "\t" in x_line
        assert not "\t" in y_line

        xy_line = "{}\t{}\n".format(x_line, y_line)

        xy_list.append(xy_line)
        indexes.append(file_offset)

        file_offset += len(xy_line)

    vocab = "".join(list(vocab))
    return xy_list, indexes, vocab

def write_file(path, file, lines):
  with open(os.path.join(path, file), "w", newline="") as f:
    f.writelines(lines)

def process_module_group(group_name, subdirs, module_name):
  module_file = module_name + ".txt"

  lines = read_files(subdirs=subdirs, module_file=module_file)
  print("total {} lines read={}".format(group_name, len(lines)))

  inputs, targets = split_into_x_y(lines)
  print("total {} samples input={} targets={}".format(group_name, 
        len(inputs), len(targets)))

  xy_list, indexes, vocab = make_jit_pairs_and_indexes(inputs, targets)
  print("{} vocab=".format(group_name), vocab)

  target_dir = os.path.join(DATASET_TARGET_DIR, module_name)
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)

  print("Writing files into {} ...".format(target_dir), end="")
  write_file(path=target_dir, file="{}.x".format(group_name), lines=inputs)
  write_file(path=target_dir, file="{}.y".format(group_name), lines=targets)

  write_file(path=target_dir, file="{}.xy".format(group_name), lines=xy_list)
  write_file(path=target_dir, file="{}.vocab".format(group_name), lines=[vocab])
  torch.save(indexes, os.path.join(target_dir, group_name + ".indexes_pt"))
  print()

def process_all_modules():
  modules = os.listdir(os.path.join(DATA_SOURCE_DIR, "interpolate"))
  print("Starting to process {} modules".format(len(modules)))
  print()

  for idx, module_file in enumerate(modules):
    module_name = module_file[:-4]

    print("{}.) Processing {} ...".format(idx, module_name))
    process_module_group("train", TRAIN_SUB_DIRS, module_name)
    process_module_group("interpolate", INTER_SUB_DIRS, module_name)
    print()

  modules = os.listdir(os.path.join(DATA_SOURCE_DIR, "extrapolate"))
  print("what")
  for idx, module_file in enumerate(modules):
    module_name = module_file[:-4]

    print("{}.) Processing {} ...".format(idx, module_name))
    process_module_group("extrapolate", EXTRA_SUB_DIRS, module_name)
    print()

print(" Done.")

if __name__ == "__main__":
    process_all_modules()