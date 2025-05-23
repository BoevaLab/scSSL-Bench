import pandas as pd 

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter

def read_data(path: str):
    return pd.read_csv(path, sep=",")

df = read_data(path='results/table_B4.csv')

datasets = list([" ".join(col.split(" ")[:-1]) for col in df.columns if col != "Method"])
methods = list(df["Method"])
df.set_index('Method', inplace=True)

# Find ranking and make
rank = []
for i, d in enumerate(datasets):
    rank.append(np.array([df.loc[m, f"{d} Pearson"] for m in methods]))
rank = np.array(rank).mean(axis=0)
tmp = np.array([[x, y]for y, x in sorted(zip(rank, methods))])
methods = list(tmp[:, 0])
rank = list(tmp[:, 1])

# make plot for each dataset
fig, axs = plt.subplots(1, 1, figsize=(23/4, 4), dpi=500)
width = 0.29
yticks = [i for i in np.arange(0.0, 1.01, 0.1)]
hatches = ["\\", ".."]
colors = ["tab:orange", "tab:gray"]
colors_hatch = ["peachpuff", "silver"]

metrics = {
    "PBMC": np.array([df.loc[m, f"{datasets[0]} Pearson"] for m in methods]),
    "BMMC": np.array([df.loc[m, f"{datasets[1]} Pearson"] for m in methods])
}
 

multiplier = 0
x = np.arange(len(methods)) - width/2

for attribute, measurement in metrics.items():
    offset = width * multiplier
    rects = axs.bar(x+offset, measurement, width-0.015, label=attribute, hatch=hatches[multiplier], color=colors[multiplier], edgecolor=(colors_hatch[multiplier], 0.5))
    multiplier += 1

axs.grid(True, "major", axis="y", ls="--", linewidth=0.4, alpha=0.8)
# axs.set_title(d, fontsize=20,)  
axs.margins(x=width/12)

axs.set_ylim([0, 1])
axs.set_yticks(yticks)
axs.tick_params(axis="y", length=0.0)

axs.set_xticks([n for n in range(len(methods))])
axs.set_xticklabels(methods, rotation=25, ha='right', fontsize=16,)
axs.tick_params(axis="x", pad=-0.06)

axs.set_xlabel("Methods", fontsize=20, loc='center') 
        
axs.set_ylabel("Pearson Mean", fontsize=20,) 
axs.set_yticklabels(sum([["{:3.1f}".format(i), ""] for i in np.arange(0.0, 1.0, 0.2)], []) + ["1.0"], fontsize=17) # love it, I wanted a one liner :)


handles, labels = axs.get_legend_handles_labels()

labels[0] = "PBMC CITE-seq"
labels[1] = "BMMC"
plt.legend(handles, labels,
           loc="upper center",
           ncol=3,
           fontsize=20,
           bbox_to_anchor=[0.46, 1.42]
           )  

plt.subplots_adjust(left=0.13, right=0.99, top=0.8, bottom=0.29, wspace=0.02, hspace=0.7)
plt.savefig("plots/tableB4.svg")