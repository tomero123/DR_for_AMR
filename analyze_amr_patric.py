import pandas as pd


file_path = "./PATRIC_genomes_AMR.txt"
all_df = pd.read_csv(file_path, sep='\t')
df = all_df[['genome_name', 'antibiotic', 'resistant_phenotype']]
null_rows = (df['genome_name'].isnull()) | (df['antibiotic'].isnull())


