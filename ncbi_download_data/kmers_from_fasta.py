from Bio import SeqIO
import json


path = "C:/University 2nd degree/Thesis/Pseudomonas Aureginosa data/"
input_folder = "genome_files"
output_folder = "kmers_files"

K = 10
kmers_dic = {}

file_name = "/GCF_004370345.1_ASM437034v1_genomic.fna"
fasta_sequences = SeqIO.parse(open(path + input_folder + file_name), 'fasta')
for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)
    for start_ind in range(len(sequence) - K + 1):
        key = sequence[start_ind:start_ind + K]
        if key in kmers_dic:
            kmers_dic[key] += 1
        else:
            kmers_dic[key] = 1

with open(path + output_folder + file_name.replace(".fna", ".txt"), 'w') as outfile:
    json.dump(kmers_dic, outfile)
