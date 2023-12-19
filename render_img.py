import os
import torch
from PIL import Image


WIDTH=448
HEIGHT=800


def group_files_by_batch_and_idx(file_names):
    grouped_files = {}
    for file_name in file_names:
        parts = file_name.split('_')
        batch_idx = parts[-2] + '_' + parts[-1].split('.')[0]
        if batch_idx not in grouped_files:
            grouped_files[batch_idx] = []
        grouped_files[batch_idx].append(file_name)
    return grouped_files

def load_and_combine_images(file_group, save_dir):
    image_matrix = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
                     ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]

    images = []
    for row in image_matrix:
        image_row = []
        for position in row:
            for file_name in file_group:
                if position in file_name:
                    image_tensor = torch.load(file_name)
                    image = Image.fromarray(image_tensor.numpy())
                    image_row.append(image)
                    break
        if len(image_row) == 3:
            images.append(image_row)

    combined_image = Image.new('RGB', (WIDTH, HEIGHT))
    y_offset = 0
    for row in images:
        x_offset = 0
        for img in row:
            combined_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += row[0].height

    combined_image.save(os.path.join(save_dir, file_group[0].replace('.pt', '.png')))


def main(file_names):
    grouped_files = group_files_by_batch_and_idx(file_names)

    save_dir = './saved_images_iou/rendered_img'
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, file_group in grouped_files.items():
        if len(file_group) == 6:
            load_and_combine_images(file_group, save_dir)
folder_path__list=['./saved_images_iou_top', './saved_images_iou_down']


for folder_path in folder_path__list:
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    main(file_names)
