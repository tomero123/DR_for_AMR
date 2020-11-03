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


class AggregationMethod(Enum):
    EMBEDDINGS = "embeddings"
    SCORES = "scores"


class ClassifierType(Enum):
    XGBOOST = "xgboost"
    KNN = "knn"


class FileType(Enum):
    GENOMIC = "genomic"
    CDS_FROM_GENOMIC = "cds_from_genomic"
    PROTEIN = "protein"
    FEATURE_TABLE = "feature_table"


# antibiotic dic
ANTIBIOTIC_DIC = {
    Bacteria.PSEUDOMONAS_AUREGINOSA.value: ['imipenem', 'ceftazidime', 'meropenem', 'levofloxacin', 'amikacin'],
    # Bacteria.PSEUDOMONAS_AUREGINOSA.value: ['imipenem'],
    Bacteria.MYCOBACTERIUM_TUBERCULOSIS.value: ['isoniazid', 'ethambutol', 'rifampin', 'streptomycin', 'pyrazinamide'],
    Bacteria.TEST.value: ['amikacin', 'levofloxacin', 'meropenem', 'ceftazidime', 'imipenem'],
}

EMBEDDING_DF_FILE_NAME = "embedding_df.h5"
METADATA_DF_FILE_NAME = "metadata_df.csv"


PREDEFINED_FEATURES_LIST = {
    Bacteria.PSEUDOMONAS_AUREGINOSA.value: {
        "imipenem": ["CGTGCCGCGC", "TGGCCTGCGT", "TCACAGGACA", "TATTGCCCTG", "ACCTACAAGA", "CCAATGTACG", "CCGGCGCGAA", "TTGCCGAAAA", "GAGGTGGCCA", "CCTGTCCGCT", "TTCTCGTTTC", "GCCCTGGATA", "TGATGGAGGC", "CGTCGCTCGC", "TCGCGGGTCA", "CCACGCCACA", "ATGGGTGCGC", "GAGCCGAGTC", "GCCAACGGAC", "GGCAGCTCGG", "TAACCGTTTG", "GGATACTTCC", "TGCGGCGTTT", "TCTACAGCGC", "TTGCTTAACT", "TAAGCACCGC", "CTGAAAGCCC", "TACGGCCCCA", "ATTTCTCTCG", "GATCTCCAGC", "AGAAGCGAAT", "GTCGAAGTCC", "CGTCACCGAC", "GCCGGAACGG", "CCCGTTTAAA", "AGCTGGATGG", "CATGGCCGCT", "AGAAGGAATC", "TACGACTACC", "GATGCGGGCA", "GCCTCATCGA", "GGGGGGGCGG", "TATCGAGATC", "GCCCGCTTGG", "CTACGCCTAT", "GCTACCCCTG", "CTTGTTCCTC", "GGGTGGATGC", "GTCGCTGTAC", "GTCCTCCTCC", "AATCGGTGAC", "GACCTGCCTG", "CGTCGACCGT", "CGCAGTTGCA", "CTCGCCGCAG", "GATGAAGAAG", "TAGACCTGCA", "TCACGCTTCT", "TTCTTCGGCG", "ATAACTAACG", "CCGACAAGGG", "GACGTACGCA", "AGGCCTTTTT", "CCTCGCGCAC", "TGACGCGCCC", "GTGGCCGTGT", "CGCAGCTTTC", "TGGCCAATCG", "GATGGGATTT", "TCGCCCAGCC", "CAACCGCGTC", "CGGTGCCCGG", "GATCGAGACT", "AAACCCGGCG", "ACTGCTGCGA", "TTGTGTTGAG", "GAAGGTGACT", "CGCACGCAAG", "GCTTCCAGGG", "ACGACCACCA", "CTCGTCGGTG", "TGTTCGACAC", "TGGGCGAGCA", "GCGAGAGCTA", "CGGAACGGGC", "CACGGAGCTG", "CAGTGACCTT", "AGAAACGAAT", "AGATTTCTCT", "TCCGGCTCAC", "GTGGCCCGTG", "TGAGGGGAAT", "CCCGCCGGCC", "GATCAGCCAG", "AGCGCGCTGG", "TCGATGCCCG", "CGCCCAGGTG", "GTTTATTATA", "ACTTTGCCCG", "TGGTGCCACT", "GTTGCGCGGA", "GAACTGCTCT", "GAGTGCGCTG", "CTGGCCGCGT", "GTACTGGTCC", "CGCAGCGTGA", "GGTGGCTTTC", "GGAAGGCATC", "AGTTTGGATC", "GCCACGTATT", "ATCGGTGTCC", "GCGAAACCCC", "ACTCGGCATC", "GATGCTCCCT", "CGCAAGGCCG", "CAGTGGGGAC", "TCAAAATCGA", "CTTCTGACGA", "TTCAATGCCA", "CCCGGCCGGC", "GTGGGAATGC", "ACGCAGCGCC", "GCCGACATCG", "CGCCGCAGTC", "TAGGTCAGAG", "GTTTAACTCG", "TGCCCTACAG", "GCTTCCTGCC", "CGCGTTCCAT", "CTGGGTGTTA", "ATCAGTTGGG", "TCCGGTGCGG", "GGATGTGCAG", "CCGGTGAAAG", "ACTCGCCGTT", "AGAACCTGTC", "CACCCACGCC", "AATAGAAGCC", "GGCAGCGAGG", "CTTGCTGCCA", "TTGCCGAAGG", "CTCTTTCATA", "AAACCTGTCG", "GATCAGCGTA", "CCGAGGAGGT", "CATTTGAAGC", "CAGCGAGCCA", "AAAAGCGCAG", "ACAGCCATTC", "CTGCGGCATC", "TCAATGCCAC", "GGCGTAGCCG", "CACAACGCCA", "TAGCGGCGCT", "TACGGGGTCA", "TTGCCGTCCC", "TTCAGCAGCG", "TTTGCCGCCT", "CAAAGGTGGC", "GCCTCTTTTT", "TTCTCCTATT", "AGGCAATCAT", "GAGCGTCATT", "AGGTGCGCCA", "TAACGATCCG", "CGCCGGAGTG", "CAACACCTTG", "GGCCGGCAAT", "CCACGGTGAC", "GCCCCGCCGG", "CCAGACGCGC", "CCTGTGAATA", "GTCTTCTTCG", "GGTCGAGCGG", "CGGCGAGGTT", "GCTCAACTGC", "GCACGTTACC", "TGTGGCAGGA", "AAGGGATCGG", "CGAGACACGG", "ACCACGCGGT", "GCGCCGCAGC", "TGATCGACAG", "ACAGTACGTA", "TCCAGTGGCC", "CTTCTGACAA", "GTTCTTTGAG", "TGAGCACATC", "AGCACCTCGG", "TCTGGTCAAG", "GATCAGGTTC", "TATCCAGTTC", "TCTTTTATGA", "AACTCCTGGC", "CAACGCTGAG", "CGTAAGATCG", "CCATTCCCGA", "CACCAGCAGG", "CCGAATTCAT", "GATGGTCGTA", "ATCGCGGGGT", "TAGCACCGTT", "GGATCAGCGA", "TCCGCCTGAA", "GACCAGTCCA", "CCACTGGTCC", "CGGCTTGACC", "GAGAAAGAAC", "CCATCGATAG", "CGACTCATGC", "GGGTCGCGGA", "CTGCGCAAAT", "ATCACATCGT", "GCATCAACTG", "CTGGCGAATT", "CTAATGATGG", "CCTCCGCAGT", "AGGCGACGGT", "TCGCATAGGT", "CCCGGCATGG", "ATTCCGCACA", "CTCGCGAGCA", "ATCGACAACA", "AACATGGTGG", "CATGACCATG", "GGCTATCGTC", "CAAGCCTTTG", "ATCAGCGAGG", "AGTGTTCTCG", "ACCTGGTGGT", "GCGGACGTAA", "TGTTGATGCA", "TGCGACCACC", "GCACATCCGC", "GCAGCCGGTG", "AGCAGGACCG", "ATCAGTCGAA", "CAGAAGACGG", "CGTCCGCGCG", "AGTCGGTGTA", "AGCCTCTATC", "GCCTATGCAC", "CGGCACGAAC", "TTCGCTGGCG", "GATCGGGCAT", "CCCCATGATG", "TGCCGACAGC", "AATGGCATAA", "CTCAGCACCT", "AGTTGCTGGA", "CGGTCTTGTT", "ACCCCGCGCC", "CAAAGCATAT", "GTGATAGTCC", "CTTCGAGCGT", "TGATCATGGC", "TGACGGGTGC", "TTTCCAACAT", "CCGTTCAACG", "GTAGCTTGGA", "CTGATCCACG", "GAACACCACC", "TCCGGCGGGC", "CGAAGCAGAG", "GCCCGGACGC", "TGGTGACCTC", "GGGTTATATT", "TTTGGCAATC", "CTGGAATTTT", "ATCGATAATC", "TCGATCCGGC", "TCCTGGTACA", "AAGCTGGCCG", "GACGTGCCGC", "GGTAAGCATC", "GTTTGCCCCT", "GCGATTTGCT", "GCTATAGTGC", "ACATGTCCCG", "TCGCCAATTT", "CGCTTTCCCC", "CGAAGGTTCG", "CAGTACATGT", "CGCAGTAGGT", "CAGGCTTGAA", "TAGTTGCAAC", "CAATGGCACC", "GTTGTCTCGC", "TTGGCGAATG", "GTGCGGCCGC", "TTCATCAGAG", "CGGACACCCC", "CAAAGCTTAA", "GTCGTCGACC", "GACAGACCTG", "CTCCGCCTGC", "GGCCGACGGC", "TTCCTCTTTC", "ACAGCAATCA", "CGCGGCGAAG"],
        "ceftazidime": ["GGGAGATATA", "CGACTCCACT", "TGCTCGGCGG", "TCCGCGCTAT", "CTCGCAAGGA", "TACGGCCGAA", "TCGAGGCGGG", "TCCCAATAGT", "CCGCCCACAG", "ATTTCTACCG", "CATCCGGAGG", "AAAATACCCA", "TGAAAGGGGT", "CGTCCCAAGA", "GACCCCGTTT", "GCGAAAACCG", "CCTTCTCGTT", "TCGTAGCCCA", "TTGTGGTAGT", "CATCCGACCG", "AGGATGGTAC", "CTATCGGCTT", "TGGTGTCGAT", "GCCGGCATAG", "TTGCACGCCG", "CGGAAGTTAT", "GCCCAGTGCT", "CGCTACATGA", "CTTAAGTACA", "GTACCGTTTT", "AAATCTTCCA", "AAACCAAGTA", "GCACCGCCAA", "GACAAGATGG", "CCCTGGCTGG", "GTAACTGGTT", "CATCATGTGT", "CGTCCCAAAT", "ACGGCGCGCC", "CATCACGGGG", "CCTGGTTGGA", "GCCCGCTTCG", "CGTGGCGGTA", "GAATCTGAGC", "TCACATTAAT", "GGACAGGTGA", "GACCTGCTGG", "AATGTGCAGA", "CGCCTTAGTT", "TTCAACTACG", "TGTTAATGAG", "ATGGCCTGCG", "AGCTCGGCGG", "GTTTTCTGCA", "CCGAATCTCC", "CCTGCTGCCG", "CAGGTACGCC", "ATCGACCTCG", "TGGCGGCCGG", "AAGGTGCTAT", "AACGTCCCAC", "GAGCCTGCGC", "TATTCGTAGC", "GACGTAACTG", "GGCAACCGCT", "AGGGCGTACT", "TCTTCAACGT", "GGAATTTTGT", "GAGAGTTCCT", "GTCTACACCG", "GCGGGCGTGC", "GCTCGGCGTG", "TCAAGAGCGG", "TGCTCGGCGA", "AGCTTCGGCG", "TCCAAGTATT", "TCGTTTCAGC", "CGCCCAGATC", "AACTTCATGA", "CCTCGTTTTG", "GGTTATTCAT", "CCCTCTCCCA", "TCAACAACTA", "GCCAAGGAAA", "TCGCATTGAA", "TATGAATACT", "TGCAGTTCCG", "GTCGGCTTCT", "GTTATTCATC", "GTCCGAGTTC", "TCGGGCACGA", "CCTGCCCGCA", "GGAGTCCATG", "GGCGACGCTC", "TGGCTGGACC", "GGACCTCCGC", "GTGAGACAAA", "ATGGGGCGGG", "GGAGTCGATT", "TGCGGCCTCT", "CGGTGCATTC", "CCATGTCACT", "GGCGCTCATA", "TGACGAAGAT", "TTCGCAACCA", "ATGCACGACG", "GATACAGCTC", "ACGCAGACTT", "TGGGTGATTG", "AAACCGAGGG", "GCGATGACGA", "AAGCCTTGCA", "GGGCGAATAG", "CACAAGACGG", "TGCTGCATCA", "ATCGGGCGCG", "TTCACGCGGC", "TGGTTGCGAT", "TCCCGCCGAA", "ATGAGCCTGT", "CCTCGCGTGA", "GAGAGATTGA", "CCTTTTAAGG", "GATAGCTCGC", "TGGGGCCTGT", "TTATGCGTTG", "CTTACCTGCA", "CGTTCTCGGT", "CCGGCCACGC", "GGGCGCTCCT", "GCGCTCGCCG", "GCACACTCCA", "GAACTCGCGC", "GGCTCGGGTC", "GCCTTTGGCT", "AATCGTCGTT", "ATGCGCTGTA", "CAGATAGCAA", "AAGAAGGTAA", "AGGCGTGGAC", "CAAAGGTGCC", "CATGCAACTG", "TCTGAGTCAG", "TCTGTAACAG", "GCGCCACCAA", "GAGCGATTGT", "TCAAGTTTAA", "CCGAGCCGGT", "TTACTCCGTC", "GGCTTGACCA", "CAGAAGCGAG", "AAGGTCTGAA", "CCGGGGCTGG", "GCAGTGACCA", "GCGTGCCGGC", "GTTGCCGTGC", "TGACCCGAGC", "GGCGTCGCCC", "CTCGGATTAC", "TAGGCGACAA", "CCAGCAGCGA", "GTTCGTCCTC", "AGAAATCACG", "CTTTCTACTT", "AGATGCGGCT", "CCTGGGTCAC", "CAACGACTGC", "GCCTGGTCAT", "GATCGGTATT", "GGGCAGTATA", "AGTACTTTGC", "CCGACCTCAA", "ACAGGCTCTT", "CAGACCGGTG", "TGAACAGCCT", "GCCGCTGCGC", "CGAGCTGGAC", "ACTGACTGAC", "TAGCAGGAAT", "GCATCTAGAT", "TCCCTCTCCT", "AGGATTGCAG", "CGCCGAGCAG", "TCGGAGGTCG", "GTCGAACAGT", "ACGACCAACA", "TCTCGCGTCT", "GTGCAACAAG", "AACTGCAGAA", "AGGCTCGGGT", "TCCCCGGTAC", "TGCGCGTTGT", "ATTGCCCTGG", "CGACGCACTT", "TATCGCGGGT", "AGAACTCACC", "GGTTTTTTGG", "TCAAGATTTG", "TTGCACCCAC", "GTGTGATGGG", "TGGCTTTCAG", "AATATGGCGT", "GGATTCGCAC", "ACCCCAAATA", "AAACTGTGTA", "CGTTTAAAGA", "CCCGCGTAGT", "GTGCCCGACT", "ATAGCGACGA", "AAAGGGAGGC", "CCTGGGCGGC", "GCCTGGTGCA", "TCTGGGGAAC", "CAGGCGGCCC", "TCTAGCTTCG", "CTCGTGGATC", "GGGTGCCGGA", "ACCAGCCAGA", "CGGCGGGGTG", "ACCAAGGCGC", "TCCAGCTCGC", "CGTTCCTCTC", "GGCGCCTCCG", "GCTCCACCCG", "ACTCTACATG", "AGCTTGGATA", "GTATCGAAGA", "CAAAGCTTCT", "CATAAGGAAA", "CGGTGGACGC", "GAACGGCGGC", "GCGAATTCTA", "TCCAGATTGG", "TCCTGAGCGA", "AGGGCTTTGG", "CTTCGTTCCT", "AAGGCACCCT", "CGGCATCATT", "TTTGCAGAAG", "CTGCCCTATG", "TTTCGAAGTG", "ACGTGCCATT", "CCTATCCCAG", "TGACGTCCTT", "CTTCGCCGAT", "GCGCACATAG", "AGCAACGGGC", "CTTCGGATAT", "GAATGGCGTA", "GCAGTAGCTC", "ATCGAGATGT", "GACCATGCGG", "CGCGATGTAG", "AGCGACCCCA", "GAGTCCTACC", "GGGGCAGTCC", "ACTTGGTCCT", "GTGGCTCCGG", "TGGTCGCGAA", "CCGTCGAACA", "GCGATCGAGA", "ATGCGGAGAA", "CAAAATGGTG", "TTTACCTGCA", "TGGACTCATC", "AACCACTTCT", "TTACCGGCAA", "GGTCCTTGTA", "GGCAACGCCT", "AAAGGCGACT", "ATGGTCGACC", "AGACTGATAC", "CATTCCGACC", "AAGGGATGAT", "GGCGTCCACT", "CATCGGTCAG", "GCTGTCGAGG", "CGGTTAATAC", "GATTACTCCT", "TTCATTACGA", "CTACGGTCGA", "CTGCGAATCA", "GACTGTCTTC", "ACGATGCAGG", "AGTAACTTTT", "GGATGCTAAT", "TGGCGGCTGT", "GGTGCGCGAG", "CTTCCTATAA", "CGACCCCTGA", "CGGTTCGGTC", "CGTCCTCCCG", "TATAGCAGCG", "ACTCATGGGC", "CGCGAATCAC", "CCAGCTTCAT", "CCCAATAGTG", "CTAGCATCGA", "CAGCACCCAG", "GAGATGGCGC"],
        "meropenem": ["CCTTGGATAT", "TCACTAGTGA", "CTCACTAGTG", "TGTCTTACCG", "GAAATGCTGT", "ATGGTGAGGC", "CGGGGGGTTA", "TGGCGAGGCA", "CCAGCGCTGG", "CTTGAACTCG", "TTGCAATTCC", "CGAGCAAGAA", "TATTTTTTCG", "GCGGCGCGAT", "GCCTATTTGA", "GTCTTGAGCG", "AGGGCACTGA", "GATGCAACCC", "GAGATCTACA", "CACTTTTCGA", "CTTGACGTTC", "CCCGTGCACG", "CAATGAAAGT", "GGTTAGGCCG", "CCCCACGGGT", "GGGGTACCGT", "GTGCGTCATC", "CCGCCGCGGC", "GCCAGAATAG", "TGATTTCCGC", "TCCCGCGGCC", "CCTGGCGATT", "TGCCTGCGGG", "TTGCACACAG", "GGGGATGCGC", "CTTTTCGATT", "CGGAGCAGAA", "ATGCACGACG", "CCCGGGAAGG", "AAGGGCTTTG", "TGTCCGGCGA", "AGTGGCCTTC", "TTGCCATGCG", "TAGGAGCTCG", "ACCAGCCGCC", "CCTAAATCGG", "CTTCGCCAAC", "TCAACGCCAA", "GTTTAATTTT", "CAGACACGAA", "AATCGGATGC", "CCTCGAGACG", "ATTGCTTGGC", "CCGTGTTCTG", "CGGGGAGCTG", "CGAACCCAGG", "TCGCCTGTGA", "CCTCGACCTG", "CTGACCTATT", "ATTGTTCTCT", "TCGCAGCGCG", "AGTTTTCTAT", "CTCGCGCAGG", "CTGTGCACGG", "CCGGAAGACC", "GGAACCGCGC", "TGTGCGTGAA", "ACCTACTGAT", "CTTCTGCTCG", "GCGTGTATTG", "TGTAAAGGAT", "GCCAGCCACG", "TGATGGCATG", "GCCTGGCTGG", "CGAAGGCGAT", "TACGGTGTGT", "GGATGATTCC", "GCGGAGTTTC", "TCCGCATACA", "TAAAGCCTGA", "GCTCTACCAA", "TGTCCGTGGA", "TTGTTCTGCG", "CGTAGTCGGG", "GCAACATGGC", "AAAGCGAGCG", "TCACAAAAAA", "TCGCCGATGG", "GCGCTGGCGA", "AGCCCATGGG", "AACCCCAAAG", "GTCCAGCCTG", "TGACGCACGA", "TGTGCGTCAC", "TGCTGAGTCG", "CCAAGAAAAG", "GGTTGGCTTA", "CTGAGGACCG", "TGCTGGTTCT", "CTTATCCGAC", "ATATCCCTGA", "ATCGACATCG", "GGAGACGATA", "GATCACCCGT", "CAAGCCTCAG", "GTGACGAGGA", "TACGATGCAT", "TCGATCAACC", "CAACACAGGT", "CGCTGCGACG", "CAGGCTAAAG", "GCCAGCCTCT", "ACCGCGATGT", "GCAGGTAGAT", "GATGCACCAC", "TTGGGCGGAT", "TGACGACGAT", "GTCATGCCCT", "GACAACCTTT", "CTCCGTCCCG", "ACCTGCAGAA", "TGTCTCTTTG", "TTCAGGAGGG", "CTGAATATGA", "AACGGTCCGT", "GGACTGGGCC", "ATCAACAATC", "TTGACCTGTA", "AACTGGTGCG", "GACAACACTA", "TTACAACACT", "TGTCCGACAA", "ACGAATGAAC", "TAAACTTTTC", "AGGCGGTGAC", "TCCTGCTGGT", "CGCGGACCTT", "TAACTTTGCT", "CCGGACCTCT", "GGCTGTCGAC", "CAGCGCTAGC", "GATGGCATAG", "ACGCCGTCGG", "ATCCAAGTTC", "CTGAGCGTTT", "GGCTTGTCCG", "CCGCATTTGA", "TTCTTCGGTC", "CCCAGGAGAA", "TCTCGCTATT", "CAGGACCCGA", "ACAACTGCCC", "CACGGAGCAG", "CTTGTCCTTA", "CCTTTCCATA", "GCCGCGGTGC", "GATCCGATCT", "TTTCCGAGGT", "CGACGCTGAT", "GGCCCAGGGT", "TACATTACCG", "TCTTGTAAAC", "AGCAACTCCG", "GGGTTAAATG", "GCCTGGCTAC", "ATAGAGGTTT", "GAACTCCTAG", "CCATACGACG", "CGCGCGTCAC", "GAAAAGGACC", "AAGGAAAGTC", "CCGCTGTAGA", "AATCGCCGCT", "AGGCAGCCAC", "CCCACTGGAC", "CGGCTGGGAC", "GCGTGTACGT", "AACCCTTCGA", "GGCTACTTCA", "TGTACCTGGA", "GTCTATTAGT", "AGCTTGCGGC", "ACTCGGAGGA", "CTCCGCCATA", "GACGAGAGTT", "GTCCAAAGCC", "CTGCTATTCG", "GCCCCCTGGC", "AATGGAAAAA", "TGACCATGAA", "TTTCATCTCT", "CAGCACACGA", "GTCCATTCTG", "ACGTGTGATT", "CACCCCATGA", "CCGGGCGACC", "ACCCCCGGGG", "AGTCCACCAA", "TAGGCCGACC", "TGACTGGATT", "CTGCAACTAA", "TCGAGGAAGT", "CGCCGGCCGA", "GATCCCGTGG", "TGCGTCGTTA", "GTGGCACCAA", "GTCCTGGGCC", "GTTTTGGGAT", "ACGACATCAG", "TTCAACTGGT", "TCTCCAGGAA", "GAACTCCCGG", "ACAGACAGGA", "GTCTAGCACC", "GACGCGCATC", "GCTTGTTGCT", "CGAATGTCAC", "GTTTAAGCCT", "CCTCGCTCTT", "AGCGTGAACG", "GCAGCTACGT", "CGCAGCTCCC", "TTAAGCATCT", "TTTCGAAAAT", "TATCTACTCT", "ATGCAAGTAG", "CGGACCTGGG", "TTGGACCATT", "GCGCACAGGC", "GTGCATCGTG", "AGCGTTTTTA", "ATGGCACGTT", "TCCGTGATCC", "GGCATGCATA", "TGACTACGCG", "GCGCAGTGCG", "CCACCCCCGT", "TCGCGGCAAT", "CCCACTTCGG", "CATGCAGCGG", "CCAGCCCCAT", "TGCCTACATC", "TGACCATCCC", "GGAGCGCCGC", "CGGGCTCGAC", "AAGAATCAGT", "GTGGATATGA", "ACCCTAGCCG", "GCCTGATGGA", "TTCCACGCCG", "GCTCGGAGAA", "CTGGACTACC", "GCTTTATCGT", "ACTCGTGGCC", "TCACCGAGAC", "CGCTCTCCTC", "GGCGTTCGAT", "GGACTGGCGT", "ACGGGCCCGT", "GGTGACCGGC", "CCCTGCGGGA", "TCGGTTACAG", "TCTTGCGCAG", "TCAGGCGACT", "CAACCAGATC", "TCGAAACTGA", "CCCCCCAGCA", "GCGCTTCTAT", "CTGCACCCTA", "TCCGTGCGCT", "CCCGCACCTT", "ACGATCAGCC", "CCATGCGGTC", "TGCGAGGAGC", "CCCAGAGCAA", "CGATGTGGTC", "CGGGTTTCTC", "TCTGCGCACT", "TTGAGAAATG", "TAGTGAGTAA", "AGGCACCGGG", "GTGGGTGAGA", "TCAACATCAT", "GTTGGTGGCT", "CTGGCGCCGT", "GTAATAGGCC", "GCAACCCAAC", "GGGCGACGTA", "TCTTCATGAA", "CTCCCAGCGG", "GACAGGCTGT", "GTGCCTACAT", "GGTGAGCTTA", "AGCTCCATTG", "CCCTTCGTCC", "GCCAACACGT", "CTGGATCCAG", "CGGTTTTCGT", "AGGAATTTCA", "AAGAAAATCC"],
        "levofloxacin": ["AATCGGTGTC", "TCACTAGTGA", "AGGAGGCGAG", "GTCCTCGTAG", "TTACTATACT", "ACACGAGTAA", "GTTCTTGCTA", "TGTCTTCGGA", "TCGGACCCGG", "CGCGATGGGC", "CTGGACGCGA", "GATATGAGGG", "CAATTGCGGG", "TGTGATGACG", "ACGAGTGATA", "TACTCGGGAT", "GGACATTCAA", "ATAATTTTTA", "GCCTCTACTC", "CTCGCTCATT", "GCCTGGGAAA", "GCGAATACCT", "ATTTGCGTGA", "GCCGTACTTG", "GCCTACGGCG", "GTGAAGGTGA", "CGTTTTTCGA", "AATAGCCTAG", "CCTGGTTTGG", "CGGCGAAGTT", "CCACCGGTGT", "CAACAGGCCC", "AAGGTACCTT", "CTACCAGCAC", "ATCGGGTTTT", "ACGGCCATCG", "GATGCCGATC", "GGCAAGATGC", "CGCGGCTCTA", "TACGAGATGG", "ATAGTGCAGT", "GCGTCGATGA", "ACGACTCGTA", "CCCCAGTAGG", "CACGGCCACT", "TATCCAGGTC", "CTGGTCGATG", "CTTTGTCGAA", "GTCTGGGCTG", "ATGCCACCGG", "TCTAATAAGG", "TTCTTGCCTG", "TACCAGCAAT", "CGCCTCATCG", "CAGCAAGGAG", "TGCTTTATCC", "GGTCGCCCAT", "GCAGCTATGG", "AAGATACCTT", "CGTTCTGTCC", "TACTGCGCAG", "CTTGGATGCA", "CGCTGGTCAC", "GACAGTGGCA", "GCTCGAACGT", "TCTCAATAGG", "TAAGAAGCGG", "GCGCGCGTCC", "CCACAGCAGT", "ACGGTGTGAG", "GGACAAAGCA", "GTTGATTCAT", "AGCCCGCCTT", "CAACTTGCTG", "CTAGACTCAG", "ACCGTAACTT", "CAGCAGTCGA", "GGACACCACG", "CCTGCTCAGC", "GCAATTGCTT", "AAGCACCACT", "CAGCGAGTCC", "CCGCTGCAGG", "TGAGCGTTAT", "GGCATCATCG", "CTTTTGCGAG", "GATCGGTATT", "ATCTGCGCGC", "ATATCGTAAA", "CATACCATAT", "ACTGCATACC", "GACGCTCCGT", "ACCAGGGTGG", "CGCTCTTAGA", "CAACTCCGCC", "CGCGATGGCC", "CGCAGGAAGC", "TGTCCGGCAG", "GGGGGGACGT", "TCGACACAGA", "CGTGACTTGG", "GCGCGACCTA", "CACCGGACAG", "CAGCTTGCCA", "GATGATCCGC", "ACTTGAGCAT", "TGGGCGTTGG", "TTTTAGGAAG", "GGCTTTAACT", "TTGAGCCGGG", "AGGCCGCTGA", "GTGGTCCGGC", "TCGCATGCCC", "GTTCTATAAC", "GTCTTCCGTG", "ACGCAATTTG", "CACGGGCAAG", "CAGTATCAGC", "AGTCAAATAC", "CGGCGGCCAT", "CCTTGAGGAC", "CCTAGATCTG", "ACCGTGCGCC", "ATTTTAAAAT", "AAAATACAGT", "CGACACCGCA", "CCAACCTGCG", "CATCAGTAGC", "AGCAGGCCAT", "ATGGCCAAGA", "TTTGGGATTG", "GTAGTTCGCC", "AAGAGCCGCG", "CACACCTGAT", "TTTGGAAATT", "TGTAGACGCC", "CTTGCGTGCC", "AGGCGCTGCG", "CGAGGGGGTA", "GAGGTATGAC", "AGATAGTCGT", "CAGATTGAAT", "ACGGGTAGTA", "GGATCCCGCG", "TAGCGGCGCG", "TCGGCGATGC", "GAGAGCGTCT", "GGTCAGCACT", "ACTCAGTCAG", "GATGTCGGGG", "CAGGGATGCG", "TACATCGAAA", "GAGCTGACCA", "GAAGGTGGGC", "TGCGGTATCG", "CTTTTGAAGC", "CGCCCTAAAA", "CTCTACGTGA", "TCGGTTTCCA", "AGCGGTCGCT", "GGCACTTCCA", "TACCCTAGGC", "GCTGCAACAT", "GGACACGAGC", "TATGCCGAAA", "GCAGGGCCGT", "AGACGAGCAA", "TATGGAAGGT", "CATCCGGCTG", "CCGCGGAGAC", "AACTTCCCCA", "GTTGCCTCTG", "CCACCACCAA", "GCGTGCAGAA", "GTAAGCGAGG", "CTCACTTTTT", "ACCTGATAGG", "AGCAGCAACT", "AACGGCCGCC", "CGTATTAAGT", "TCGAGATTGG", "TGCGATCACG", "CGAACAGTCG", "GGAATGCCAA", "GACGCCAGCC", "GCGCCTCTAA", "TGATTTTTCG", "ACACCTGGCG", "CCACCTGCTG", "ACGTTCAAGG", "CCGCGATGTC", "GGTTTCAAAG", "CTGTGGCTAA", "TACCCATTCG", "CGCCGACTAG", "CTGCTGACCG", "GCACTGCTGG", "GAACCGATAA", "ATCCGCGACG", "TTCGCGCTGC", "CCTCGGCACG", "TTTGCAGTAG", "CTCTGGTGTG", "GCGCTGGTAA", "AATCTCCGTG", "GAATATCTTC", "TTGCATCGTG", "CGTATTAACC", "ACAAGCTAAC", "TTGCAATTCC", "GTGCGGCTGC", "CTGCAGGACA", "TACCAAGGCG", "CAGCATCATT", "AGTCCTATGG", "GAGGAAGGCC", "TCTTTAGAGA", "CCGGATACCA", "GCCCCGGCTC", "AGGGCGGGAT", "ATTGAACATG", "GCTGGTGAGT", "CGCCTGACAA", "AGGAAATTCC", "AAAAACCCCG", "GCCGAGGTCA", "GGTCGCTTGC", "TGCTCACTGC", "CCCTGAGCAT", "CCAGACCTGC", "TGAAGGCCTC", "CGTCGAACCG", "CACGATTGAC", "TTCCACCTAG", "CTTTAGCCTG", "GGGTCTGCGG", "ATGGTTGTCG", "CATGGACCCA", "GCGGGCGATG", "CGAGCATGCG", "AAAGAAAGAA", "GTCCGAACTC", "GCATTTATTT", "CAAGATATTC", "GTGTTTTTTC", "GTTCTTTGAC", "ACGCTTCTTA", "ACATCGACAC", "CCTTGCCGGC", "TCAGTTAGAC", "AAGCAGCGCA", "CCGAGCCGGT", "GGGAGCCGAA", "TAGCCGTCGA", "GTTAGTAGAG", "AAAGTGGTTA", "AGTCTTCTGA", "GATGCGCGGA", "ACGTACGCGC", "CGACCATGGA", "GCACGCCCAC", "ATATAGGTGC", "AAACGCCCCA", "CTGGCGCACA", "GAGCAGCTGC", "TCGTATAGGC", "TATCAGAGAA", "GACCGGCAAC", "TCGAGGTCTA", "TTCACTGACC", "CTTGTGATAC", "ATATCTATAT", "GTGAGAATCA", "CAGGCATTCA", "CTCTATCCCT", "ATCGACTATG", "ATTTGGCCGA", "AGCATCTGCC", "CAGGGCCCGG", "CCCCACATTT", "CGTTCTTCCT", "GTTTTCTAAA", "CTCAAGAACC", "TCAAGATCTA", "AGGACAGAAA", "GACGTATGCC", "GGTCTACCAC", "AGTCTGAGTA", "CCCTACACGG", "CAGGGCGGCA", "GATGGACAGA", "ATCGTCGAGC", "ACAACGTGAC", "ATTCGCATGG", "CGGTTGCTTG", "CATTAGGCAT", "CATAGGAGTT", "GCTTGGCCAC", "TTGCATTTCT", "GGTTTCAAGG"],
        "amikacin": ["CTCCAGGTAT", "TCACTAGTGA", "CGCCCCCCCC", "GCATCTCGCA", "TCTGCAGAGT", "TGCAGACAAC", "ACCTGGAGAC", "ACAGACGCAA", "TCAAACCACG", "GAAGTGTACA", "ACGCCAACGG", "AGAAGGCCGG", "AAGGCCTCCC", "CCAAGAGTTC", "CCAGGGGGGG", "TAGAGCGCGA", "GCCTAACACA", "TATAGGCGCC", "GGTGCTTCTC", "CCAACGTGCT", "TCGGCGCGAT", "TGGGCCGAGA", "CAATCACTTG", "CGATGAGAGC", "CGGATCTGAG", "TGCTGTCGAT", "CCGAGCCTGG", "GGTGTAGTGC", "CCCCGGCGTC", "ACGCCCTGGG", "AAAAAGAGCA", "AGATCCTCCG", "TTGCGACCTC", "GCAGCCAGGC", "ACCACCTATA", "AACCGCCCAT", "TTGCGAACTG", "CGTCAATCTC", "ATTATGTGAT", "ACAGCGCCCT", "CTCTCAGTTA", "TTATGGTGCC", "GTGGGCAGGA", "GTTTTAAAAC", "GGAAAGTGTT", "TCAGCGTGAG", "TACCAACAAG", "CTGGTTCGAG", "GGGCTCCTAC", "TACAGCCTTG", "CTTCGTAGCC", "GACCCGGACT", "GGCTCACGGA", "AGCGAGAAAT", "GGCGTGCCGA", "TTGGCGACAG", "ATGTTGATTA", "AAGCGACTGC", "CGCCGGGCTT", "CAGGGGATAG", "CTTCTATCAC", "GCGAGGGCCC", "TGCCCACCAC", "ATCGAATCCA", "TATTATCTTG", "AGCAGATATC", "TTTACCTGGG", "AATTAAATTT", "GCCGCTGCGC", "CCCTCAACCA", "ATGGCAACCG", "GTTCTCTCCG", "TTATCCCACT", "TTTATGGGCG", "GGTAGGCGAG", "TTCGCCACTA", "CCTTGGGACT", "TCCCGACGCG", "CGCACCGGCT", "TCGACGTCGA", "TTGGAGAAAC", "GTGGAACGCG", "ACTTGATGGG", "AGCCTCGTTA", "ATACTCGTCC", "CCGTTGAGAA", "GAACGCACAT", "TTTTTATCAC", "CTGTCTACCG", "CAACATTACT", "GTCGCTGAAG", "CTCTACTCAA", "TCGAACAAGC", "CATCGTATTG", "GCCTAGCCTA", "GATTGAGCTC", "GCTAGTGAAA", "GCCCTTGCCG", "TCCGCCGCCG", "GTTTAGGGGT", "GCTACAACAT", "GTCGCGGTCC", "TCGCTCTTAT", "AGGCGCTTAC", "AACCGCTGCC", "GGCCCTCGAA", "GTTGACCCAT", "CAAGATTGAC", "CCACGGCCTG", "GCCGACGGCA", "ACGCCGCTCA", "AAGTAGGCTC", "TCAGCCACTA", "CTGTCTGTAC", "TGATCGAGGA", "CTTTTCCGGA", "TGCCCTGGGT", "CCATATTTGT", "ATCCGGCTGA", "TATCGCCGCC", "GATGTTTGAT", "AAGCGAAGGA", "ATCGTCCTCT", "GTCTTGGCAC", "ATCTCCCTTT", "GCCGACGAGA", "CGGGTGGGTC", "CAATCAGGCT", "TGTTCATCCT", "TCAACACAAG", "TTTGCAACAG", "TACCCTCTTG", "TCCCGGCCTT", "ATAACGGACT", "AACGGGCTAT", "CATTGGCTAA", "CCCACTGATG", "TGAAGTCGCA", "ACGCGCAATA", "GACAACTGGA", "AGTTGACCTG", "TCATCCCGTG", "ACAACTCCGC", "CGATGGTGGC", "CAATATCAAC", "CCGCCTGGGC", "CTGCTGCGAC", "ATTCTTACGT", "CGTAGAGACG", "TCGTCTGAAG", "CTCAGACTGG", "GCATCCGGCT", "GATCCGCCGG", "ACCATAACTA", "AGTACGCGAG", "GGTAGCCGAT", "CAGCCTTGCA", "AGGCTACTGG", "GAGGCCCAGG", "CCCAGCAACC", "CAACCGACTT", "TTGTCGTCGT", "TACGCCGCGA", "GGGAAGGCGA", "TTAAAGAGTG", "CATATAGCCC", "GAGCACCACC", "AGCACGCTCG", "AACGGAAGTA", "GCGCACCTGG", "GATGTTGTGA", "ACTACACCAA", "AGTGGACCTC", "CTACTTCTGC", "TCGCTTAAAC", "TTCGCTCCTA", "GGTGCGTTTG", "GGCTGTGTTT", "CCAAAACGCC", "GTTCTGATGA", "CCCCGACCTG", "CTCGCCCTTC", "GGACGAAGTG", "GGCCCTCGAC", "ACTAGCGGCC", "TCAGCGGCGG", "CAATATCTAT", "CCATCAGCTC", "TTCAACTTCT", "AGGGGGCGAT", "GCGGCACGGA", "GTGTTGCTGA", "TGCTGGTTAG", "GCGCTAACGC", "GTGGGTTTGC", "GCATCAGGTT", "TGGATATTCG", "GGCAGATTCG", "CCTTCTCGCC", "AGGTGTACTC", "GAATGTGTAA", "CGCAGAGAAA", "ATCCTTTACA", "CCCTCTTTTA", "CAAGGTCTAC", "ACTACTCGGA", "GGTTACACGA", "GGTACCGATG", "AGACGATATG", "CGCGACTGCG", "AACAAATCGC", "CCTCCGCTTC", "GAGCAAGAGT", "GAGTGCCCGC", "CGACCGCTCG", "TGGTTACGGG", "AGCTCGGATC", "AGTATCTCCA", "GGTGCTGCCG", "GTCTTTCGGC", "CGCACAGGCG", "AAAAAGGATT", "AAGTTCGGTG", "GCGTTCTTCC", "CTACCTTGTC", "GCATCAGCCT", "CGGCGCATCG", "CAGGCTGTCC", "GCCATGAAGT", "ATTCATCGTA", "CGGCTTGACC", "TTGAGTACGC", "GGAGGAGCGT", "TAACTGGCTC", "GATGCCATCG", "TTGGCCTCGA", "GAGGTGGTTA", "ACCTCCATGG", "CTATCGACGC", "GCCTAACAAT", "GCCTGTCAAT", "GGATCGCTCT", "CAATCACCTG", "CCGGATCGCC", "CACGATCCCT", "CACCCGACAC", "TAGAAGCTGT", "TGGCGCCCGG", "TCCGTACCGC", "GTTTGCGTTT", "CGGGGATAGA", "TAGCGGCTGT", "CCGCAGCCAA", "TCGGCTTCCC", "GCTCACTCAA", "AGGCCCAAAT", "GTGCTCGGAC", "GCGCCTGAGT", "ATTTTCCCAA", "GCTGGCCAGT", "GTTCATCCTA", "TCCGTTCAGT", "CTTGGTGTGG", "CAGGTGTCCG", "CCTTGCCTAC", "CGAGACCCAG", "TAGACTCCGC", "TTCGAGTAGC", "TTACTTGAGG", "AAGATGTGTT", "AGCGGCTGCT", "AGCCTACCCG", "CATTCCAGAA", "CCCCCCCCTC", "GGTTGACCGA", "CCCGACCCTG", "CAAGCCTCGT", "GCCGGCCATG", "TAGTCCATGG", "ATCGGCGACG", "AGCCGGCGCC", "TGGCTCGACG", "TGGTGCCACC", "TAGTCCGCGT", "GCCTTCTTCG", "GGTGAGGTAG", "GAGGGTAAGC", "GGTTAGCGGT", "ATTGTTAGGC", "GGCAGGGGTT", "CCACCACGCT", "TACTGCGCGA", "TCACCGTCGC", "CTTTTGTTTA", "ATGCTCCGGG", "GACGCGAGCC", "GCAGCGTATA", "CCAGTTTGTG", "ACTGGTCGCG", "GTTCGTGGTC"],
    }
}

FILES_SUFFIX = {
    FileType.GENOMIC.value: ".fna.gz",
    FileType.CDS_FROM_GENOMIC.value: ".fna.gz",
    FileType.FEATURE_TABLE.value: ".txt.gz",
    FileType.PROTEIN.value: ".faa.gz",
}

TIME_STR = '%Y-%m-%d %H:%M:%S'
