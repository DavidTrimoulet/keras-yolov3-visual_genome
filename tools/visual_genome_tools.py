# tools to convert visual genome data to yoloV3

from pathlib import Path
import json
import cv2


def convert_object_for_yolo_v3(path= Path('.'), filename="objects.json"):
    densecap_file = path / filename
    yolo_file = path / "yolo_objects"
    id_to_object_file = path / "yolo_object_to_id"
    object_to_id = {}
    object_id=0
    with open(str(densecap_file), 'r') as df:
        with open(str(yolo_file), 'w+') as yf:
            data = json.load(df)
            image_count = 0
            for image in data:
                image_count += 1
                if image_count < 5000 :
                    image_id = image["image_id"]
                    images_path = path / "images" / "VG_100K"
                    image_annotation = '{}/{}.jpg'.format(images_path, image_id)
                    for image_object in image["objects"]:
                        cur_object_id = 0
                        x_min = image_object["x"]
                        y_min = image_object["y"]
                        x_max = image_object["x"] + image_object["w"]
                        y_max = image_object["y"] + image_object["h"]
                        object_names = image_object["names"][0].replace('.', ' ').replace('/', ' ').replace(';', ' ').replace('\'', ' ').replace('\"', ' ').replace(',', ' ').replace('\0', ' ').split(' ')
                        for object_name in object_names:
                            if object_name not in object_to_id:
                                object_to_id[object_name] = object_id
                                cur_object_id = object_id
                                object_id += 1
                            else:
                                cur_object_id = object_to_id[object_name]
                            object_string = '{},{},{},{},{}'.format(x_min, y_min, x_max, y_max, cur_object_id)
                            image_annotation = '{} {}'.format(image_annotation, object_string)
                    image_annotation = '{}\n'.format(image_annotation)
                    yf.write(image_annotation)
    print(len(object_to_id))
    object_to_id_list = ["" for i in range(0, len(object_to_id))]
    for key, value in object_to_id.items():
        object_to_id_list[value] = key

    with open(str(id_to_object_file), 'w+') as itof:
        for object_name in object_to_id_list:
            itof.write('{}\n'.format(object_name))


def convert_object_for_retina(path= Path('.'), filename="objects.json"):
    densecap_file = path / filename
    yolo_file = path / "retina_objects"
    id_to_object_file = path / "retina_object_to_id"
    object_to_id = {}
    object_id=0
    with open(str(densecap_file), 'r') as df:
        with open(str(yolo_file), 'w+') as yf:
            data = json.load(df)
            image_count = 0
            for image in data:
                image_count += 1
                if image_count < 5000 :
                    image_id = image["image_id"]
                    image_path = path / 'images/VG_100K/{}.jpg'.format(image_id)
                    for image_object in image["objects"]:
                        cur_object_id = 0
                        x_min = image_object["x"]
                        y_min = image_object["y"]
                        x_max = image_object["x"] + image_object["w"]
                        y_max = image_object["y"] + image_object["h"]
                        if y_min >= y_max:
                            y_max = y_min + 1
                        if x_min >= x_max:
                            x_max = x_min + 1
                        object_names = image_object["names"][0].replace('.', ' ').replace('/', ' ').replace(';', ' ').replace('\'', ' ').replace('\"', ' ').replace(',', ' ').replace('\0', ' ').split(' ')
                        for object_name in object_names :
                            if object_name not in object_to_id:
                                object_to_id[object_name] = object_id
                                cur_object_id = object_id
                                object_id += 1
                            else:
                                cur_object_id = object_to_id[object_name]
                            object_string = '{},{},{},{},{}'.format(x_min, y_min, x_max, y_max, object_name)
                            image_annotation = '{},{}'.format(image_path, object_string)
                            image_annotation = '{}\n'.format(image_annotation)
                            yf.write(image_annotation)
    print(len(object_to_id))
    with open(str(id_to_object_file), 'w+') as itof:
        for key, value in object_to_id.items():
            itof.write('{},{}\n'.format(str(key), str(value)))


def display_image_bb(path= Path('.'), filename="yolo_objects", id_mapping="yolo_object_to_id"):
    yolo_file = path / filename
    id_mapping_file = path / id_mapping
    id_to_object = []
    with open(str(id_mapping_file), 'r') as m:
        for line in m:
            id_to_object.append(line)
    #print(id_to_object)
    with open(str(yolo_file), 'r') as f:
        data = f.readline()
        data = data.split()
        image_file = path / "images" / "VG_100K" / '{}.jpg'.format(data[0])
        print(image_file)
        img = cv2.imread( str(image_file), cv2.IMREAD_UNCHANGED)
        for i in range(1, len(data)):
            object = data[i].split(',')
            cv2.rectangle(img, (int(object[0]), int(object[1])), (int(object[2]), int(object[3])), (0, 255, 0), 5)
            cv2.putText(img, id_to_object[int(object[4])], (int(object[0]), int(object[1]) + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
