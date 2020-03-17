import gzip
import json
import seaborn as sns
import matplotlib.pyplot as plt

with gzip.open("../results_files/all_kmers_file.txt.gz", 'rt') as f:
    all_kmers_dic = json.loads(f.read())

print(f"Total number of kmers: {len(all_kmers_dic)}\n")
strains_count_list = []
hist_dic = {}
for count_list in all_kmers_dic.values():
    strains_count = len([x for x in count_list if x > 0])
    strains_count_list.append(strains_count)
    if strains_count in hist_dic:
        hist_dic[strains_count] += 1
    else:
        hist_dic[strains_count] = 1

for k in sorted(hist_dic.keys()):
    print(f"Strains Count: {k}; Number of kmers: {hist_dic[k]}")
# sns.distplot(strains_count_list)
# plt.show()
#
# x = 1
