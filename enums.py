from enum import Enum


class Bacteria(Enum):
    MYCOBACTERIUM_TUBERCULOSIS = "mycobacterium_tuberculosis"
    PSEUDOMONAS_AUREGINOSA = "pseudomonas_aureginosa"
    GENOME_MIX = "genome_mix"


class ProcessingMode(Enum):
    NON_OVERLAPPING = "non_overlapping"
    OVERLAPPING = "overlapping"
