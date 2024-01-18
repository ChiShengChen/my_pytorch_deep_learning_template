import os
import shutil

def process_image_file(txt_file, source_dir, dest_dir):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        rel_path, _ = line.split()
        folder_name, image_name = os.path.split(rel_path)

        source_path = os.path.join(source_dir, folder_name, image_name)
        dest_path = os.path.join(dest_dir, folder_name)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        shutil.copy(source_path, dest_path)

# Example usage
txt_file = '/home/meow/my_data_disk_5T/food_classification/CNFOOD-241/test_n.txt'  # Update this with the path to your text file
source_dir = '/home/meow/my_data_disk_5T/food_classification/CNFOOD-241/train600x600'    # Update this with your source directory path
dest_dir = '/home/meow/my_data_disk_5T/food_classification/CNFOOD-241/for_CUB_200_2011_format/test'        # Update this with your destination directory path

process_image_file(txt_file, source_dir, dest_dir)
