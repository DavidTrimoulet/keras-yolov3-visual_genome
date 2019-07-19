# tools to convert visual genome data to yoloV3
import random
from pathlib import Path
import json
import cv2
import re


class VisualGenomeTools:

    def __init__(self, path_to_visual_genome_folder=Path('.'), load_glove=False):
        self.path = path_to_visual_genome_folder
        self.data = []
        if load_glove:
            self.glove_vocab = self.load_glove()
        self.dataset_vocab = {}

    def set_visual_genome_data(self, data):
        self.data = data

    def load_visual_genome_data(self, filename="objects.json"):
        vg_file = self.path / filename
        with open(str(vg_file), 'r') as vgf:
            self.data = json.load(vgf)
        print("data loaded : ", len(self.data), "elements")

    def get_glove_vocab(self):
        if not self.glove_vocab:
            self.load_glove()
        return self.glove_vocab

    def get_dataset_vocab(self):
        return self.dataset_vocab

    def clean_visual_genome_object_data(self):
        print(len(self.data))
        self.remove_not_in_glove_words()
        self.remove_less_used_words()
        self.remove_single_character_from_vocab()
        self.remove_plural()
        self.set_vocab_to_the_n_most_used_word()
        self.re_indexed_vocab()
        # self.split_vocabulary()
        self.clean_dataset_with_dataset_vocab()
        print("final vocab word count:", len(self.dataset_vocab))
        print(len(self.data))

    def re_indexed_vocab(self):
        re_indexed_vocab = {}
        index = 0
        for key, value in self.get_dataset_vocab().items():
            re_indexed_vocab[key] = index
            index += 1
        self.dataset_vocab = re_indexed_vocab

    def save_data_and_vocab(self, output_data, output_vocab):
        vg_clean_file = self.path / output_data
        vg_clean_vocab = self.path / output_vocab
        with open(str(vg_clean_file), 'w+') as vgf:
            json.dump(self.data, vgf)
        with open(str(vg_clean_vocab), 'w+') as vgf:
            json.dump(self.dataset_vocab, vgf)

    def remove_plural(self):
        print("len dataset vocab before reduction of plural words :", len(self.dataset_vocab))
        new_dataset_vocab = {}
        for vocab in self.dataset_vocab:
            if vocab[-1] == 's':
                if vocab[0:-1] not in self.dataset_vocab.keys():
                    if vocab[-2] == 'e' and vocab[0:-2] not in self.dataset_vocab.keys():
                        #print(vocab, "seems to not be a plural")
                        new_dataset_vocab[vocab] = self.dataset_vocab[vocab]
            else :
                new_dataset_vocab[vocab] = self.dataset_vocab[vocab]
        print("len dataset vocab after reduction of plural words :", len(new_dataset_vocab))
        self.dataset_vocab = new_dataset_vocab

    def replace_by_singular(self, word):
        if len(word) > 0:
            if word[-1] == 's':
                if word[0:-1] in self.dataset_vocab:
                    #print("replacing ", word, "by", word[0:-1])
                    return word[0:-1]
                else:
                    if len(word) > 2:
                        if word[-2] == 'e' and word[0:-2] in self.dataset_vocab:
                            # print("replacing ", word, "by", word[0:-2])
                            return word[0:-2]
        return word

    def remove_less_used_words(self, thresold=5):
        print("len dataset vocab before reduction of less used word :", len(self.dataset_vocab))
        new_dataset_vocab = {}
        for vocab in self.dataset_vocab:
            if self.dataset_vocab[vocab] > thresold:
                new_dataset_vocab[vocab] = self.dataset_vocab[vocab]
        print("len dataset vocab after reduction of less used word :", len(new_dataset_vocab))
        self.dataset_vocab = new_dataset_vocab

    def remove_single_character_from_vocab(self):
        print("len dataset vocab before reduction :", len(self.dataset_vocab))
        new_dataset_vocab = {}
        for vocab in self.dataset_vocab:
            if len(vocab) > 1:
                new_dataset_vocab[vocab] = self.dataset_vocab[vocab]
        print("len dataset vocab after reduction :", len(new_dataset_vocab))
        self.dataset_vocab = new_dataset_vocab

    def remove_not_in_glove_words(self):
        print("len dataset vocab before reduction with glove vocab :", len(self.dataset_vocab))
        new_dataset_vocab = {}
        glove_vocab = self.load_glove()
        for vocab in self.dataset_vocab:
            if vocab in glove_vocab:
                new_dataset_vocab[vocab] = self.dataset_vocab[vocab]
        print("len dataset vocab after reduction with glove vocab  :", len(new_dataset_vocab))
        self.dataset_vocab = new_dataset_vocab

    def clean_dataset_with_dataset_vocab(self):
        data_with_dataset_vocab = []
        for image in self.data:
            # print(image)
            updated_image = image.copy()
            updated_image["objects"] = []
            # print(image["objects"])
            for image_object in image["objects"]:
                updated_image_object = image_object.copy()
                text = re.sub('[^A-Za-z]+', ' ', image_object["names"][0].lower())
                # print(image_object["names"][0])
                words = text.split(" ")
                clean_words = []
                #print(words)
                for word in words:
                    word = self.replace_by_singular(word)
                    if word in self.dataset_vocab:
                        clean_words.append(word)
                    #else:
                        #print("word not in vocab:", word)
                if clean_words:
                    updated_image_object["names"] = [" ".join(clean_words)]
                    updated_image["objects"].append(updated_image_object)
            # print(updated_image["objects"])
            data_with_dataset_vocab.append(updated_image)
        self.data = data_with_dataset_vocab

    def generate_vg_object_vocab(self):
        # Dictionary with { word :  occurence }
        dataset_vocab = {}
        for image in self.data:
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
        self.dataset_vocab = dataset_vocab

    def generate_vg_region_vocab(self):
        # Dictionary with { word :  occurence }
        dataset_vocab = {}
        for image in self.data:
            # print(image)
            for image_region in image["regions"]:
                text = re.sub('[^A-Za-z]+', ' ', image_region["phrase"].lower())
                words = text.split(" ")
                for word in words:
                    if word in dataset_vocab:
                        # si le mot est déjà dans le vocabulaire, on increment l'occurence
                        dataset_vocab[word] += 1
                    else:
                        # si le mot n'est pas dans le vocabulaire
                        dataset_vocab[word] = 1
        print("generated dataset vocab length :", len(dataset_vocab))
        self.dataset_vocab = dataset_vocab

    def set_vocab_to_the_n_most_used_word(self, n=1000):
        sorted_by_occurrence_vocab = [(k, self.get_dataset_vocab()[k]) for k in sorted( self.get_dataset_vocab(), key=self.get_dataset_vocab().get, reverse=True)]
        print("len dataset vocab before reduction to", n, "elements:", len(self.dataset_vocab))
        new_dataset_vocab = {}
        for i in range(n):
            new_dataset_vocab[sorted_by_occurrence_vocab[i][0]] = sorted_by_occurrence_vocab[i][1]
        print("len dataset vocab after reduction to", n, "elements:", len(new_dataset_vocab))
        self.dataset_vocab = new_dataset_vocab

    def convert_object_for_yolo_v3(self):
        yolo_object_file = self.path / "yolo_objects"
        id_to_object_file = self.path / "yolo_object_to_id"
        with open(str(yolo_object_file), 'w+') as yf:
            for image in self.data:
                image_id = image["image_id"]
                images_path = Path() / ".." / "Visual_Genome" / "images" / "VG_100K"
                image_annotation = '{}/{}.jpg'.format(images_path, image_id)
                for image_object in image["objects"]:
                    x_min = image_object["x"]
                    y_min = image_object["y"]
                    x_max = image_object["x"] + image_object["w"]
                    y_max = image_object["y"] + image_object["h"]
                    words = image_object["names"][0].split(" ")
                    for word in words:
                        cur_object_id = self.dataset_vocab[word]
                        object_string = '{},{},{},{},{}'.format(x_min, y_min, x_max, y_max, cur_object_id)
                        image_annotation = '{} {}'.format(image_annotation, object_string)
                image_annotation = '{}\n'.format(image_annotation)
                yf.write(image_annotation)

        print("final vocab length :", len(self.dataset_vocab))
        with open(str(id_to_object_file), 'w+') as itof:
            for object_name in self.dataset_vocab:
                itof.write('{}\n'.format(object_name))

    def convert_region_for_captionner(self):
        yolo_region_file = self.path / "region_pairs"
        region_vocab = self.path / "region_vocab"
        number_of_region = 0
        number_of_valid_region = 0
        with open(str(yolo_region_file), 'w+') as yf:
            yf.write("image_path,caption\n")
            for image in self.data:
                #print(image)
                for region in image["regions"]:
                    number_of_region += 1
                    image_id = region["image_id"]
                    images_path = Path() / ".." / "Visual_Genome" / "images" / "VG_100K"
                    image_annotation = '{}/{}.jpg'.format(images_path, image_id)
                    x_min = region["x"]
                    y_min = region["y"]
                    w = region["width"]
                    h = region["height"]
                    sentence = region["phrase"].lower().replace(',', '')
                    region_string = '{} {} {} {} {}'.format(x_min, y_min, h, w, sentence)
                    image_annotation = '{},{}'.format(image_annotation, region_string)
                    image_annotation = '{}\n'.format(image_annotation)
                    all_word_in_vocab = True
                    text = re.sub('[^A-Za-z]+', ' ', sentence)
                    words = text.split(" ")
                    for word in words :
                        if word not in self.dataset_vocab:
                            all_word_in_vocab = False
                    if all_word_in_vocab:
                        number_of_valid_region += 1
                        yf.write(image_annotation)
        print("Number of region:", number_of_region, ", number of valid region:", number_of_valid_region)
        print("final vocab length :", len(self.dataset_vocab))
        with open(str(region_vocab), 'w+') as itof:
            for object_name in self.dataset_vocab:
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

    def convert_pairs_for_seq2seq(self, filename="region_pairs", input_train_filename="input_train", output_train_filename="output_train", input_dev_filename="input_dev", output_dev_filename="output_dev", input_test_filename="input_test", output_test_filename="output_test", ):
        file_path = self.path / filename
        input_train_filepath = self.path / input_train_filename
        output_train_filepath = self.path / output_train_filename
        input_dev_filepath = self.path / input_dev_filename
        output_dev_filepath = self.path / output_dev_filename
        input_test_filepath = self.path / input_test_filename
        output_test_filepath = self.path / output_test_filename
        input_data = []
        output_data = []
        with open(file_path, 'r') as fp:
            data = fp.readlines()
            for line in data:
                values = line.split(",")
                input_data.append(values[0])
                output_data.append(values[1:])
        n_pairs = len(data)
        train_size = int(0.8 * n_pairs)
        dev_and_test_size = int(0.1 * n_pairs)
        print("pairs =", n_pairs, ", train =", train_size, ", dev =", dev_and_test_size)
        print("generating training set")
        self.save_dataset_part_to_file(input_data, input_train_filepath, output_data, output_train_filepath, train_size)
        print("generating dev set")
        self.save_dataset_part_to_file(input_data, input_dev_filepath,
                                       output_data, output_dev_filepath, dev_and_test_size)
        print("generating test set")
        self.save_dataset_part_to_file(input_data, input_test_filepath,
                                       output_data, output_test_filepath, dev_and_test_size)
        print(len(input_data), len(output_data))

    @staticmethod
    def save_dataset_part_to_file(input_data, input_train_filepath,
                                  output_data, output_train_filepath, train_size):
        with open(input_train_filepath, 'w+') as itf:
            with open(output_train_filepath, 'w+') as otf:
                data_size = len(input_data)
                for i in range(train_size):
                    val = random.randint(0, data_size)
                    itf.write(input_data.pop(val))
                    otf.write(" ".join(output_data.pop(val)))
                    data_size -= 1
