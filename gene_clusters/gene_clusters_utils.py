import gzip
import os
import re
import shutil
from functools import partial

import pandas as pd
from Bio import SeqIO

from constants import FileType, FILES_SUFFIX


def extract_file(file):
    with gzip.open(file, "rb") as compressed:
        with open(file[:-3], "wb") as uncompressed:
            shutil.copyfileobj(compressed, uncompressed)


def create_merged_file(input_list):
    ind = input_list[0]
    file = input_list[1]
    input_folder_base = input_list[2]
    output_folder = input_list[3]

    features_table_file_path = os.path.join(input_folder_base, f"{FileType.FEATURE_TABLE.value}_files", file + "_" + FileType.FEATURE_TABLE.value + FILES_SUFFIX.get(FileType.FEATURE_TABLE.value))
    protein_file_path = os.path.join(input_folder_base, f"{FileType.PROTEIN.value}_files", file + "_" + FileType.PROTEIN.value + FILES_SUFFIX.get(FileType.PROTEIN.value))
    cds_file_path = os.path.join(input_folder_base, f"{FileType.CDS_FROM_GENOMIC.value}_files", file + "_" + FileType.CDS_FROM_GENOMIC.value + FILES_SUFFIX.get(FileType.CDS_FROM_GENOMIC.value))

    output_path = os.path.join(output_folder, file + ".csv.gz")

    features_data = pd.read_csv(features_table_file_path, compression='gzip', delimiter='\t')

    # genes
    gene = features_data[((features_data["# feature"] == 'gene') & (features_data["class"] == 'protein_coding'))]
    cds = features_data[((features_data["# feature"] == 'CDS') & (features_data["class"] == 'with_protein'))]
    gene_join = gene.merge(cds, on='locus_tag')

    # pseudogenes
    # pseudo_gene = features_data[((features_data["# feature"] == 'gene') & (features_data["class"] == 'pseudogene'))]
    # pseudo_cds = features_data[((features_data["# feature"] == 'CDS') & (features_data["class"] == 'without_protein'))]
    # pseudo_join = pseudo_gene.merge(pseudo_cds, on='locus_tag')

    # proteins
    protein_df = pd.DataFrame(columns=['id', 'protein'])
    extract_file(protein_file_path)
    for record in SeqIO.parse(protein_file_path[:-3], "fasta"):
        protein_df.loc[len(protein_df)] = [record.id, record.seq]

    # cds
    dna_df = pd.DataFrame(columns=['locus_tag', 'dna'])
    extract_file(cds_file_path)
    for record in SeqIO.parse(cds_file_path[:-3], "fasta"):
        try:
            locus_tag = re.findall('\[locus_tag=(.+?)\]', record.description)[0]
        except AttributeError:
            locus_tag = ''
        dna_df.loc[len(dna_df)] = [locus_tag, record.seq]

    # create genes merged file
    gene_join = gene_join.merge(dna_df, on='locus_tag')
    gene_join = gene_join.merge(protein_df, left_on='product_accession_y', right_on='id')
    gene_join.to_csv(output_path, compression='gzip')

    print(f"Finished processing file #{ind + 1}: {file}")

    # create pseudogenes merged file
    # pseudo_join = pseudo_join.merge(dna_df, on='locus_tag')
    # pseudo_join.to_csv("pseudo_genes.csv")
