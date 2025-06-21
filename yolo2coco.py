def yolo_to_coco(yolo_dir, output_json_path, class_names):
    import os
    import json
    from PIL import Image
    from tqdm import tqdm

    image_dir = os.path.join(yolo_dir, 'images')
    label_dir = os.path.join(yolo_dir, 'labels')

    categories = [{'id': i, 'name': name} for i, name in enumerate(class_names)]
    images = []
    annotations = []
    ann_id = 1

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    for img_id, img_file in enumerate(tqdm(image_files)):
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

        if not os.path.exists(label_path):
            continue

        with Image.open(img_path) as img:
            width, height = img.size

        images.append({
            'file_name': img_file,
            'id': img_id,
            'width': width,
            'height': height
        })

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, xc, yc, w, h = map(float, parts)
            x = (xc - w / 2) * width
            y = (yc - h / 2) * height
            bbox_width = w * width
            bbox_height = h * height

            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': int(class_id),
                'bbox': [x, y, bbox_width, bbox_height],
                'area': bbox_width * bbox_height,
                'iscrowd': 0
            })
            ann_id += 1

    #output directories
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    coco_dict = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(output_json_path, 'w') as f:
        json.dump(coco_dict, f, indent=2)

    print(f'Done. Saved to {output_json_path}')


# call for test train val 
yolo_to_coco(
    yolo_dir='/content/drive/MyDrive/CV-Proje/visdrone_subset_10/VisDrone2019-VID-train',
    output_json_path='/content/drive/MyDrive/CV-Proje/coco_data/annotations/train.json',
    class_names=class_names
)


yolo_to_coco(
    yolo_dir='/content/drive/MyDrive/CV-Proje/visdrone_subset_10/VisDrone2019-VID-val',
    output_json_path='/content/drive/MyDrive/CV-Proje/coco_data/annotations/val.json',
    class_names=class_names
)

yolo_to_coco(
    yolo_dir='/content/drive/MyDrive/CV-Proje/visdrone_subset_10/VisDrone2019-VID-test',
    output_json_path='/content/drive/MyDrive/CV-Proje/coco_data/annotations/test.json',
    class_names=class_names
)