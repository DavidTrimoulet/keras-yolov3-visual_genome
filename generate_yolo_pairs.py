from pathlib import Path
from tools import visual_genome_tools

if __name__ == '__main__':
    p = Path('.')
    p = p / ".." / "Visual_Genome"
    vgt = visual_genome_tools.VisualGenomeTools(p, load_glove=True, filename="objects.json")
    vgt.clean_visual_genome_object_data()
    print(len(vgt.get_dataset_vocab()))
    vgt.convert_object_for_yolo_v3()
