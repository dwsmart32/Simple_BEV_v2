def compare_iou(true_file_path, false_file_path, output_top_file_path, output_down_file_path):
    with open(true_file_path, 'r') as file:
        true_content = file.readlines()

    with open(false_file_path, 'r') as file:
        false_content = file.readlines()

    iou_differences = []

    for true_line, false_line in zip(true_content, false_content):
        true_parts = true_line.strip().split(', ')
        false_parts = false_line.strip().split(', ')

        true_batch_num, true_idx, true_iou = int(true_parts[0]), int(true_parts[1]), float(true_parts[2])
        false_batch_num, false_idx, false_iou = int(false_parts[0]), int(false_parts[1]), float(false_parts[2])

        if true_batch_num == false_batch_num and true_idx == false_idx:
            iou_difference = false_iou - true_iou # camera - camera+radar, if positive => radar make it worse
            iou_differences.append((true_batch_num, true_idx, iou_difference))

    # Sort by IOU difference
    iou_differences_sorted = sorted(iou_differences, key=lambda x: x[2], reverse=True)

    # Top 50 differences
    top_50_differences = iou_differences_sorted[:50]

    # Bottom 50 differences
    bottom_50_differences = iou_differences_sorted[-50:]

    # Write top 50 differences
    with open(output_top_file_path, 'w') as file:
        for item in top_50_differences:
            file.write(f'{item[0]}, {item[1]}, {item[2]}\n')

    # Write bottom 50 differences
    with open(output_down_file_path, 'w') as file:
        for item in bottom_50_differences:
            file.write(f'{item[0]}, {item[1]}, {item[2]}\n')

if __name__ == '__main__':
    compare_iou('/gallery_uffizi/dongwook.lee/simple_bev/IOU_record_True.txt',
                '/gallery_uffizi/dongwook.lee/simple_bev/IOU_record_False.txt',
                'IOU_output_top.txt',
                'IOU_output_down.txt')
