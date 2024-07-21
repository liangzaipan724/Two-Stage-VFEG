import os
import random



def split_train_test():
    vfhq_dataset_path = "/data/VFHQ"
    sub_name_list = os.listdir(vfhq_dataset_path)
    sub_name_list.sort()
    random.seed(1234)
    random.shuffle(sub_name_list)
    tr_sub_name_list = sub_name_list[:26]
    tr_sub_name_list.sort()
    te_sub_name_list = sub_name_list[26:]
    te_sub_name_list.sort()


if __name__ == "__main__":
    split_train_test()
