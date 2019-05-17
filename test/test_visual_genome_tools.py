from unittest import TestCase
from tools import visual_genome_tools
from pathlib import Path

class TestVisualGenomeTools(TestCase):


    def test_convert_object(self):
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        visual_genome_tools.convert_object( p.absolute(), "objects.json" )

    def test_image_concerter_santiy_check(self):
        p = Path('.')
        p = p / ".." / ".." / "Visual_Genome"
        visual_genome_tools.display_image_bb(p, "yolo_objects", "yolo_object_to_id")