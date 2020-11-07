import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
from constants import Bacteria


# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]


prefix = '..' if os.name == 'nt' else '.'
files_path = os.path.join(prefix, "results_files", BACTERIA, "accessory_cds_from_genomic_files")
files_list = os.listdir(files_path)
for file in files_list:
    new_name = file.replace(".fna.gz", "_cds_from_genomic.fna.gz")
    os.rename(os.path.join(files_path, file), os.path.join(files_path, new_name))
