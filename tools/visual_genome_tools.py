# tools to convert visual genome data to yoloV3

from pathlib import Path
import json
import cv2
import re


class VisualGenomeTools:

    def __init__(self, load_glove=False, path_to_visual_genome_folder=Path('.')):
        self.path = path_to_visual_genome_folder
        if load_glove:
            self.vocab = self.load_glove()

    def get_vocab(self):
        if not self.vocab :
            self.load_glove()
        return self.vocab

    def clean_visual_genome_data(self, filename="objects.json"):
        vg_file = self.path / filename
        vg_clean_file = self.path / "clean_objects.json"
        clean_data = []
        dataset_vocab = {}
        object_id=0
        with open(str(vg_file), 'r') as vgf:
            data = json.load(vgf)
            for image in data:
                #print(image)
                updated_image = image.copy()
                updated_image["objects"] = []
                for image_object in image["objects"]:
                    #print(image_object)
                    updated_image_object = image_object.copy()
                    text = re.sub('[^A-Za-z]+', ' ', image_object["names"][0].lower())
                    #print(image_object["names"][0])
                    words = text.split(" ")
                    updated_image_object["names"] = words
                    for word in words:
                        if word in dataset_vocab:
                        #si le mot est déjà dans le vocabulaire, on increment l'occurence
                            (cur_object_id, occurence) = dataset_vocab[word]
                            occurence += 1
                            dataset_vocab[word] = (cur_object_id, occurence)
                        else:
                        # si le mot n'est pas dans le vocabulaire
                            dataset_vocab[word] = (object_id, 1)
                            cur_object_id = object_id
                            object_id += 1
                    updated_image["objects"].append(updated_image_object)
                clean_data.append(updated_image)
            for image in clean_data:
                #print(image)
                updated_image = image.copy()
                updated_image["objects"] = []
                for image_object in image["objects"]:
                    #print(image_object)
                    updated_image_object = image_object.copy()
                    text = re.sub('[^A-Za-z]+', ' ', image_object["names"][0].lower())
                    #print(image_object["names"][0])
                    words = text.split(" ")
                    isSplitted = False
                    for word in words:
                        for vocab_word in dataset_vocab.items():
                            if vocab_word != word and vocab_word in word :
                                isSplitted = True

                            #check if is composed of

                    updated_image_object["names"] = words
                    updated_image["objects"].append(updated_image_object)
                clean_data.append(updated_image)
        with open(str(vg_clean_file), 'w+') as vgf:
            json.dump(clean_data, vgf)

    def convert_object_for_yolo_v3(self, filename="objects.json"):
        vg_file = self.path / filename
        yolo_file = self.path / "yolo_objects"
        id_to_object_file = self.path / "yolo_object_to_id"
        dataset_vocab = {}
        object_id=0
        pairs = 0
        with open(str(vg_file), 'r') as vgf:
            with open(str(yolo_file), 'w+') as yf:
                data = json.load(vgf)
                image_count = 0
                for image in data:
                    image_count += 1
                    if image_count < 500000 :
                        image_id = image["image_id"]
                        images_path = self.path / "images" / "VG_100K"
                        image_annotation = '{}/{}.jpg'.format(images_path, image_id)
                        for image_object in image["objects"]:
                            x_min = image_object["x"]
                            y_min = image_object["y"]
                            x_max = image_object["x"] + image_object["w"]
                            y_max = image_object["y"] + image_object["h"]
                            words = image_object["names"][0].split(" ")
                            for word in words:
                                if word in dataset_vocab:
                                    # si le mot est déjà dans le vocabulaire, on increment l'occurence
                                    (cur_object_id, occurence) = dataset_vocab[word]
                                    occurence += 1
                                    dataset_vocab[word] = (cur_object_id, occurence)
                                else:
                                    # si le mot n'est pas dans le vocabulaire
                                    dataset_vocab[word] = (object_id, 1)
                                    cur_object_id = object_id
                                    object_id += 1
                                object_string = '{},{},{},{},{}'.format(x_min, y_min, x_max, y_max, cur_object_id)
                                image_annotation = '{} {}'.format(image_annotation, object_string)
                        image_annotation = '{}\n'.format(image_annotation)
                        yf.write(image_annotation)
        print(len(dataset_vocab))
        object_to_id_list = ["" for i in range(0, len(dataset_vocab))]
        single_words = 0
        for key, value in dataset_vocab.items():
            if value[1] < 5:
                print(key, " : ", value[1])
                single_words +=1
            object_to_id_list[value[0]] = key
        print("Number of few occurent word: ", single_words)

        with open(str(id_to_object_file), 'w+') as itof:
            for object_name in object_to_id_list:
                itof.write('{}\n'.format(object_name))

    def convert_object_for_retina(self, filename="objects.json"):
        densecap_file = self.path / filename
        yolo_file = self.path / "retina_objects"
        id_to_object_file = self.path / "retina_object_to_id"
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
                        image_path = self.path / 'images/VG_100K/{}.jpg'.format(image_id)
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

    def check_for_plural(self, object_name, object_to_id):
        print("not implemented yet")
        return False

    def check_for_typo(self, object_name, object_to_id):
        print("not implemented yet")
        return False

    def display_image_bb(self, filename="yolo_objects", id_mapping="yolo_object_to_id"):
        yolo_file = self.path / filename
        id_mapping_file = self.path / id_mapping
        id_to_object = []
        with open(str(id_mapping_file), 'r') as m:
            for line in m:
                id_to_object.append(line)
        #print(id_to_object)
        with open(str(yolo_file), 'r') as f:
            data = f.readline()
            data = data.split()
            image_file = self.path / "images" / "VG_100K" / '{}.jpg'.format(data[0])
            print(image_file)
            img = cv2.imread( str(image_file), cv2.IMREAD_UNCHANGED)
            for i in range(1, len(data)):
                object = data[i].split(',')
                cv2.rectangle(img, (int(object[0]), int(object[1])), (int(object[2]), int(object[3])), (0, 255, 0), 5)
                cv2.putText(img, id_to_object[int(object[4])], (int(object[0]), int(object[1]) + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def load_glove(self, filename="glove.6B.50d.txt"):
        vocab = {}
        full_path = self.path / filename
        print(full_path)
        with open(str(full_path), 'r') as f:
            for line in f:
                splitted_line = line.replace('\n', '').split(' ')
                vocab[splitted_line[0]] = splitted_line[1:]

        return vocab
