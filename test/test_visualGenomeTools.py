from unittest import TestCase
from tools import visual_genome_tools
from pathlib import Path
from numpy import dot
from numpy.linalg import norm


class TestVisualGenomeTools(TestCase):

    def setUp(self):
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        #data = [{'image_id': 2351061, 'objects': [{'synsets': ['test'], 'h': 24, 'object_id': 2216167, 'merged_object_ids': [2625585], 'names': ['test'], 'w': 23, 'y': 104, 'x': 261}, {'synsets': ['candle.n.01'], 'h': 34, 'object_id': 1673639, 'merged_object_ids': [2277928, 3898819, 1914902], 'names': ['candle'], 'w': 9, 'y': 156, 'x': 369}], 'image_url': 'http://crowdfile.blob.core.chinacloudapi.cn/4615/2351061.jpg'}
        #        ,{'image_id': 2351060, 'objects': [{'synsets': [], 'h': 81, 'object_id': 3898828, 'merged_object_ids': [3898850], 'names': ['alcohol'], 'w': 93, 'y': 239, 'x': 406}, {'synsets': ['animal.n.01'], 'h': 170, 'object_id': 1902712, 'merged_object_ids': [1773058, 1918198], 'names': ['animal'], 'w': 159, 'y': 138, 'x': 157}], 'image_url': 'http://crowdfile.blob.core.chinacloudapi.cn/4615/2351060.jpg'}
        #        ]
        self.vgt = visual_genome_tools.VisualGenomeTools(p, load_glove=True, filename="objects.json")

    def test_remove_plural(self):
        word = "tests"
        self.vgt.remove_plural(word)

    def test_convert_object_yolo(self):
        self.vgt.clean_visual_genome_data()
        self.vgt.convert_object_for_yolo_v3()

    def test_convert_object_retina(self):
        self.vgt.clean_visual_genome_data()
        self.vgt.convert_object_for_retina("clean_objects.json")

    def test_load_glove(self):
        self.assertEqual(len(self.vgt.get_glove_vocab()), 400000)

    def test_glove_similarity(self):
        glove_vocab = {}
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        full_path = p / "glove.6B.50d.txt"
        with open(str(full_path), 'r') as f:
            for line in f:
                splitted_line = line.replace('\n', '').split(' ')
                glove_vocab[splitted_line[0]] = splitted_line[1:]
        print(len(glove_vocab))
        base_word = ""
        while base_word not in glove_vocab :
            input("Which word to you want to get the closest words ?")
        base_word_vector = glove_vocab[base_word]
        distances_with_base_word = {}
        for word in glove_vocab:
            #print(glove_vocab[word], base_word_vector)
            a = [float(x) for x in glove_vocab[word]]
            b = [float(x) for x in base_word_vector]
            distances_with_base_word[word] = 1 - dot(a, b)/(norm(a)*norm(b))
            print(distances_with_base_word[word])
        list_of_distances_with_base_word = [(value, key) for value, key in distances_with_base_word.items()]
        list_of_distances_with_base_word.sort(key=lambda tup: tup[1])
        print(list_of_distances_with_base_word[0:20])





