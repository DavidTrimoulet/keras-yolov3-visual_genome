from unittest import TestCase
from tools import visual_genome_tools
from pathlib import Path


class TestVisualGenomeTools(TestCase):

    def test_convert_object_yolo(self):
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        visual_genome_tools.convert_object_for_yolo_v3( p.absolute(), "objects.json" )

    def test_convert_object_retina(self):
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        visual_genome_tools.convert_object_for_retina( p.absolute(), "objects.json" )

    def test_image_converter_sanity_check(self):
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        visual_genome_tools.display_image_bb(p, "yolo_objects", "yolo_object_to_id")