import os
import random
import shutil

def read_and_count_data(txt_file, class_num):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    check_list = [i for i in range(class_num)]
    check_label = []

    for line in lines:
        check_label.append(int(line.split("/")[0]))
    # print(sorted(check_label))
    print(len(check_label))
    not_in_label_class = set(check_list) - set(check_label)

    return not_in_label_class


train_output_txt = '/home/meow/my_data_disk_5T/food_classification/CNFOOD-241/test.txt'

res = read_and_count_data(train_output_txt, 241)
print(res)