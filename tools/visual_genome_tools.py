# tools to convert visual genome data to yoloV3

from pathlib import Path
import json
import cv2
import re


class VisualGenomeTools:

    def __init__(self, load_glove=False, path_to_visual_genome_folder=Path('.')):
        self.path = path_to_visual_genome_folder
        if load_glove:
            self.glove_vocab = self.load_glove()
        self.dataset_vocab = {}

    def get_vocab(self):
        if not self.glove_vocab:
            self.load_glove()
        return self.glove_vocab

    def clean_visual_genome_data(self, filename="objects.json"):
        vg_file = self.path / filename
        vg_clean_file = self.path / "clean_objects.json"
        vg_clean_vocab = self.path / "vg_vocab"
        clean_data = []
        #Dictionnary with { word : (id, occurence) }
        with open(str(vg_file), 'r') as vgf:
            data = json.load(vgf)
            print(len(data))
            dataset_vocab = self.generate_dataset_vocab(data)
            dataset_vocab = self.remove_not_in_glove_words(dataset_vocab)
            dataset_vocab = self.remove_less_used_words(dataset_vocab)
            dataset_vocab = self.remove_single_character_from_vocab(dataset_vocab)
            #dataset_vocab = self.remove_plural(dataset_vocab)
            clean_data = self.clean_dataset_with_dataset_vocab(data, dataset_vocab)
            #clean_data = self.split_vocabulary(clean_data, dataset_vocab)
            print("final vocab word count:", len(dataset_vocab))
            print(len(clean_data))
            with open(str(vg_clean_file), 'w+') as vgf:
                json.dump(clean_data, vgf)
            with open(str(vg_clean_vocab), 'w+') as vgf:
                json.dump(dataset_vocab, vgf)

    def remove_plural(self, dataset_vocab):
        print("len dataset vocab before reduction of plural words :", len(dataset_vocab))
        new_dataset_vocab = {}
        for vocab in dataset_vocab:
            if vocab[-1] == 's':
                if vocab[0:-1] not in dataset_vocab.keys():
                    if vocab[-2] == 'e' and vocab[0:-2] not in dataset_vocab.keys():
                        #print(vocab, "seems to not be a plural")
                        new_dataset_vocab[vocab] = dataset_vocab[vocab]
            else :
                new_dataset_vocab[vocab] = dataset_vocab[vocab]
        print("len dataset vocab after reduction of plural words :", len(new_dataset_vocab))
        return new_dataset_vocab

    def remove_less_used_words(self, dataset_vocab, thresold=5):
        print("len dataset vocab before reduction of less used word :", len(dataset_vocab))
        new_dataset_vocab = {}
        for vocab in dataset_vocab:
            if dataset_vocab[vocab] > thresold:
                new_dataset_vocab[vocab] = dataset_vocab[vocab]
        print("len dataset vocab after reduction of less used word :", len(new_dataset_vocab))
        return new_dataset_vocab

    def remove_single_character_from_vocab(self, dataset_vocab):
        print("len dataset vocab before reduction :", len(dataset_vocab))
        new_dataset_vocab = {}
        for vocab in dataset_vocab:
            if len(vocab) > 1:
                new_dataset_vocab[vocab] = dataset_vocab[vocab]
        print("len dataset vocab after reduction :", len(new_dataset_vocab))
        return new_dataset_vocab

    def remove_not_in_glove_words(self, dataset_vocab):
        print("len dataset vocab before reduction with glove vocab :", len(dataset_vocab))
        new_dataset_vocab = {}
        glove_vocab = self.load_glove()
        for vocab in dataset_vocab:
            if vocab in glove_vocab:
                new_dataset_vocab[vocab] = dataset_vocab[vocab]
        print("len dataset vocab after reduction with glove vocab  :", len(new_dataset_vocab))
        return new_dataset_vocab

    def clean_dataset_with_dataset_vocab(self, data, dataset_vocab):
        data_with_dataset_vocab = []
        for image in data:
            # print(image)
            updated_image = image.copy()
            updated_image["objects"] = []
            for image_object in image["objects"]:
                print(image_object)
                updated_image_object = image_object.copy()
                text = re.sub('[^A-Za-z]+', ' ', image_object["names"][0].lower())
                # print(image_object["names"][0])
                words = text.split(" ")
                clean_words = []
                #print(words)
                for word in words:
                    if word in dataset_vocab:
                        clean_words.append(word)
                    else :
                        print("word not in vocab:", word)
                    updated_image_object["names"] = [" ".join(clean_words)]
                    updated_image["objects"].append(updated_image_object)
                print(updated_image_object["objects"])
            data_with_dataset_vocab.append(updated_image)
        return data_with_dataset_vocab

    def generate_dataset_vocab(self, data):
        #Dictionnary with { word :  occurence }
        dataset_vocab = {}
        for image in data:
            # print(image)
            for image_object in image["objects"]:
                text = re.sub('[^A-Za-z]+', ' ', image_object["names"][0].lower())
                words = text.split(" ")
                for word in words:
                    if word in dataset_vocab:
                        # si le mot est déjà dans le vocabulaire, on increment l'occurence
                        dataset_vocab[word] += 1
                    else:
                        # si le mot n'est pas dans le vocabulaire
                        dataset_vocab[word] = 1
        return dataset_vocab

    def split_vocabulary(self, data, dataset_vocab):
        clean_data = []
        for image in data:
            # print(image)
            updated_image = image.copy()
            updated_image["objects"] = []
            for image_object in image["objects"]:
                print("new image")
                # print(image_object)
                updated_image_object = image_object.copy()
                text = re.sub('[^A-Za-z]+', ' ', image_object["names"][0].lower())
                # print(image_object["names"][0])
                words = text.split(" ")
                new_words = []
                for word in words:
                    cur_splitting = [word]
                    is_splitted = True
                    # tant qu'on a subdivisé le mot
                    while (is_splitted):
                        #print(cur_splitting)
                        is_splitted = False
                        # pour tous les mot du vocabulaire
                        #print(len(dataset_vocab.keys()))
                        for vocab_word in dataset_vocab.keys():
                            #print(" vocab_word: ", vocab_word)
                            next_split = []
                            # pour tous les mot du découpage en cours
                            for split in cur_splitting:
                                #print("split:", split)
                                # si le découpage en cours contient un mot du vocabulaire
                                if vocab_word != split and vocab_word in split:
                                    print("Split and vocab word is:", vocab_word)
                                    next_split.append(split.replace(vocab_word, ''))
                                    next_split.append(vocab_word)
                        # check if is composed of
                    new_words += cur_splitting
                print("previous word :", word, ", splitted words:", new_words)
                updated_image_object["names"] = new_words
                updated_image["objects"].append(updated_image_object)
            clean_data.append(updated_image)
        return clean_data

    def convert_object_for_yolo_v3(self, filename="objects.json"):
        vg_file = self.path / filename
        yolo_file = self.path / "yolo_objects"
        id_to_object_file = self.path / "yolo_object_to_id"
        dataset_vocab = {}
        object_id = 0
        with open(str(vg_file), 'r') as vgf:
            with open(str(yolo_file), 'w+') as yf:
                data = json.load(vgf)
                for image in data:
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
                            # si le mot est déjà dans le vocabulaire, on increment l'occurence
                            if word in dataset_vocab:
                                cur_object_id = dataset_vocab[word]
                            else:
                                # si le mot n'est pas dans le vocabulaire
                                dataset_vocab[word] = object_id
                                cur_object_id = object_id
                                object_id += 1
                            object_string = '{},{},{},{},{}'.format(x_min, y_min, x_max, y_max, cur_object_id)
                            image_annotation = '{} {}'.format(image_annotation, object_string)
                image_annotation = '{}\n'.format(image_annotation)
                yf.write(image_annotation)

        print("final vocab length :", len(dataset_vocab))
        with open(str(id_to_object_file), 'w+') as itof:
            for object_name in dataset_vocab:
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
                            object_names = re.sub('[^A-Za-z]+', ' ', image_object["names"][0].lower()).split(' ')
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
            image_file = data[0]
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
