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


# antibiotic dic
ANTIBIOTIC_DIC = {
    Bacteria.PSEUDOMONAS_AUREGINOSA.value: ['amikacin', 'levofloxacin', 'meropenem', 'ceftazidime', 'imipenem'],
    Bacteria.MYCOBACTERIUM_TUBERCULOSIS.value: ['isoniazid', 'ethambutol', 'rifampin', 'streptomycin', 'pyrazinamide']
}