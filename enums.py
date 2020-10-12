from enum import Enum


class Bacteria(Enum):
    MYCOBACTERIUM_TUBERCULOSIS = "mycobacterium_tuberculosis"
    PSEUDOMONAS_AUREGINOSA = "pseudomonas_aureginosa"
    GENOME_MIX = "genome_mix"
    GENOME_MIX_NEW = "genome_mix_new"
    TEST = "test"


class ProcessingMode(Enum):
    NON_OVERLAPPING = "non_overlapping"
    OVERLAPPING = "overlapping"


class TestMethod(Enum):
    CV = "cv"
    TRAIN_TEST = "train_test"


# antibiotic dic
ANTIBIOTIC_DIC = {
    Bacteria.PSEUDOMONAS_AUREGINOSA.value: ['imipenem', 'ceftazidime', 'meropenem', 'levofloxacin', 'amikacin'],
    Bacteria.MYCOBACTERIUM_TUBERCULOSIS.value: ['isoniazid', 'ethambutol', 'rifampin', 'streptomycin', 'pyrazinamide'],
    Bacteria.TEST.value: ['amikacin', 'levofloxacin', 'meropenem', 'ceftazidime', 'imipenem'],
}

EMBEDDING_DF_FILE_NAME = "embedding_df.h5"
METADATA_DF_FILE_NAME = "metadata_df.csv"
