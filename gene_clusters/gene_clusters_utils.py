import gzip
import os
import re
import traceback
from functools import partial
import json

import pandas as pd
from Bio import SeqIO

from constants import FileType, FILES_SUFFIX

_open = partial(gzip.open, mode='rt')


def create_merged_file(input_list):
    try:
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

        # proteins
        protein_df = pd.DataFrame(columns=['id', 'protein'])
        for record in SeqIO.parse(_open(os.path.join(protein_file_path)), 'fasta'):
            protein_df.loc[len(protein_df)] = [record.id, record.seq]

        # cds
        dna_df = pd.DataFrame(columns=['locus_tag', 'dna'])
        for record in SeqIO.parse(_open(os.path.join(cds_file_path)), 'fasta'):
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

    except Exception() as e:
        print(f"ERROR at create_merged_file for: {file}, index: {ind}, message: {e}")
        traceback.print_exc()


def create_kmers_from_combined_csv(input_list):
    try:
        strain_index = input_list[0]
        file = input_list[1]
        K = input_list[2]
        input_combined_genes_path = input_list[3]
        output_all_genes_kmers = input_list[4]
        output_accessory_genes_kmers = input_list[5]
        output_accessory_cds_from_genomic_files = input_list[6]
        summary_gene_files_path = input_list[7]
        all_genes_kmers_dic = {}
        accessory_genes_kmers_dic = {}
        accessory_seq_list = []
        genes_df = pd.read_csv(os.path.join(input_combined_genes_path, file + ".csv.gz"))
        with open(os.path.join(summary_gene_files_path, "STRAINS_GENES_DICT_ACCESSORY.json"), 'r') as f:
            strains_genes_dict_accessory = json.loads(f.read())
        accessory_genes = []
        for _, genes_list in strains_genes_dict_accessory.get(str(strain_index)).items():
            accessory_genes += genes_list
        accessory_genes = list(set(accessory_genes))
        accessory_genes_count = len(accessory_genes)
        for gene_id, row in genes_df.iterrows():
            dna_sequence = row['dna']
            locus_tag = row['locus_tag']
            # Do below only for acessory genes
            if gene_id in accessory_genes:
                header = str(strain_index) + "|" + str(gene_id) + "|" + str(locus_tag)
                accessory_seq_list.append(f">{header}\n{dna_sequence}")
            for start_ind in range(len(dna_sequence) - K + 1):
                key = dna_sequence[start_ind:start_ind + K]
                if key in all_genes_kmers_dic:
                    all_genes_kmers_dic[key] += 1
                else:
                    all_genes_kmers_dic[key] = 1
                # Do below only for acessory genes
                if gene_id in accessory_genes:
                    if key in accessory_genes_kmers_dic:
                        accessory_genes_kmers_dic[key] += 1
                    else:
                        accessory_genes_kmers_dic[key] = 1

        with gzip.open(os.path.join(output_all_genes_kmers, file + ".txt.gz"), 'wt') as outfile:
            json.dump(all_genes_kmers_dic, outfile)

        with gzip.open(os.path.join(output_accessory_genes_kmers, file + ".txt.gz"), 'wt') as outfile:
            json.dump(accessory_genes_kmers_dic, outfile)

        with gzip.open(os.path.join(output_accessory_cds_from_genomic_files, file + "cds_from_genomic.fna.gz"), "wt") as outfile:
            outfile.write("\n".join(accessory_seq_list))

        print(f"FINISHED processing: {file}, accessory genes count: {accessory_genes_count}")
    except Exception as e:
        print(f"ERROR at create_kmers_file for: {file}, index: {strain_index}, message: {e}")