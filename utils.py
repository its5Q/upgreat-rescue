from PIL import Image


def scale_image(image, scale=0.5):
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    return image.resize((new_width, new_height), Image.NEAREST)


def tile_image(image, tile_size=224, intersection=0.25):
    width, height = image.size

    num_tiles_x = int((width - tile_size) / (tile_size * (1 - intersection))) + 2
    num_tiles_y = int((height - tile_size) / (tile_size * (1 - intersection))) + 2

    tiles = []

    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            left = int(i * tile_size * (1 - intersection))
            upper = int(j * tile_size * (1 - intersection))
            right = int(left + tile_size)
            lower = int(upper + tile_size)

            if right > image.width:
                left = image.width - tile_size
                right = left + tile_size

            if lower > image.height:
                upper = image.height - tile_size
                lower = upper + tile_size
            
            tile = image.crop((left, upper, right, lower))
            tile_coordinates = (left, upper, right, lower)
            tile_size_tuple = (tile_size, tile_size)
            
            tiles.append((tile_coordinates, tile_size_tuple, tile))

    return tiles


def calculate_overlap_percentage(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    overlap_percentage = intersection_area / min(bbox1_area, bbox2_area)
    return overlap_percentage

def merge_boxes(bboxes):
    merged_boxes = []
    visited = [False] * len(bboxes)

    for i in range(len(bboxes)):
        if visited[i]:
            continue

        group = [i]
        visited[i] = True

        for j in range(i + 1, len(bboxes)):
            if visited[j]:
                continue

            if calculate_overlap_percentage(bboxes[i][0], bboxes[j][0]) >= 0.75:
                group.append(j)
                visited[j] = True

        if len(group) > 1:
            # Calculate the average coordinates of the group
            avg_x1 = sum(bboxes[k][0][0] for k in group) / len(group)
            avg_y1 = sum(bboxes[k][0][1] for k in group) / len(group)
            avg_x2 = sum(bboxes[k][0][2] for k in group) / len(group)
            avg_y2 = sum(bboxes[k][0][3] for k in group) / len(group)
            avg_conf = sum(bboxes[k][1] for k in group) / len(group)
            label = 0

            merged_boxes.append(((avg_x1, avg_y1, avg_x2, avg_y2), avg_conf, label))
        else:
            merged_boxes.append(bboxes[group[0]])

    return merged_boxes