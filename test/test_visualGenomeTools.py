from unittest import TestCase
from tools import visual_genome_tools
from pathlib import Path


class TestVisualGenomeTools(TestCase):

    def setUp(self):
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        # data = [{'image_id': 2351061, 'objects': [{'synsets': ['test'], 'h': 24, 'object_id': 2216167, 'merged_object_ids': [2625585], 'names': ['test'], 'w': 23, 'y': 104, 'x': 261}, {'synsets': ['candle.n.01'], 'h': 34, 'object_id': 1673639, 'merged_object_ids': [2277928, 3898819, 1914902], 'names': ['candle'], 'w': 9, 'y': 156, 'x': 369}], 'image_url': 'http://crowdfile.blob.core.chinacloudapi.cn/4615/2351061.jpg'}
        #        ,{'image_id': 2351060, 'objects': [{'synsets': [], 'h': 81, 'object_id': 3898828, 'merged_object_ids': [3898850], 'names': ['alcohol'], 'w': 93, 'y': 239, 'x': 406}, {'synsets': ['animal.n.01'], 'h': 170, 'object_id': 1902712, 'merged_object_ids': [1773058, 1918198], 'names': ['animal'], 'w': 159, 'y': 138, 'x': 157}], 'image_url': 'http://crowdfile.blob.core.chinacloudapi.cn/4615/2351060.jpg'}
        #        ]
        self.vgt = visual_genome_tools.VisualGenomeTools(p, load_glove=True, filename="objects.json")

    def test_remove_plural(self):
        word = "tests"
        self.vgt.remove_plural(word)

    def test_convert_object_yolo(self):
        self.vgt.clean_visual_genome_data()
        print(len(self.vgt.get_dataset_vocab()))
        self.vgt.convert_object_for_yolo_v3()

    def test_convert_object_retina(self):
        self.vgt.clean_visual_genome_data()
        self.vgt.convert_object_for_retina("clean_objects.json")

    def test_load_glove(self):
        self.assertEqual(len(self.vgt.get_glove_vocab()), 400000)






