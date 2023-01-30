import numpy as np
import random
import os.path

data_dir = "../../Headspace_FLAME_registrations/"
# a = np.load("../datasets/shapenet/data/split_cars.npz")

TRAIN_PERCENT = 0.7
VAL_PERCENT = 0.1
TEST_PERCENT = 0.2

random.seed(1995)

all_files_list = []
for dir_name in os.listdir(data_dir):
    all_files_list.append(os.path.join("../Headspace_FLAME_registrations/", dir_name))

train_sum = round(len(all_files_list) * TRAIN_PERCENT)
train_set = random.sample(all_files_list, train_sum)

remaining_set = all_files_list.copy()
for train_obj in train_set:
    del remaining_set[remaining_set.index(train_obj)]
val_sum = round(len(all_files_list) * VAL_PERCENT)
val_set = random.sample(remaining_set, val_sum)

test_set = remaining_set.copy()
for val_obj in val_set:
    del test_set[test_set.index(val_obj)]

train_dict = {"train": train_set}
val_dict = {"val": val_set}
test_dict = {"test": test_set}
np.savez("../datasets/split_headspace.npz", **train_dict, **val_dict, **test_dict)
