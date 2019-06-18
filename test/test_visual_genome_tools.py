from unittest import TestCase
from tools import visual_genome_tools
from pathlib import Path


class TestVisualGenomeTools(TestCase):

    def test_convert_object_yolo(self):
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        vgt = visual_genome_tools.VisualGenomeTools(path_to_visual_genome_folder=p)
        vgt.convert_object_for_yolo_v3("objects.json" )

    def test_convert_object_retina(self):
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        vgt = visual_genome_tools.VisualGenomeTools(path_to_visual_genome_folder=p)
        visual_genome_tools.convert_object_for_retina(  "objects.json" )

    def test_image_converter_sanity_check(self):
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        vgt = visual_genome_tools.VisualGenomeTools(p)
        visual_genome_tools.display_image_bb("yolo_objects", "yolo_object_to_id")

    def test_load_glove(self):
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        vgt = visual_genome_tools.VisualGenomeTools(p, load_glove=True)

    def test_clean_visual_genome_data(self:
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        vgt = visual_genome_tools.clean_visual_genome_data(filename="objects.json")