import pandas as pd 

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter

def read_data(path: str):
    return pd.read_csv(path, sep=",")

df = read_data(path='results/table_B3.csv')

datasets = list([" ".join(col.split(" ")[:-1]) for col in df.columns if col != "Method" and col.endswith("MacroF1")])
methods = list(df["Method"])
df.set_index('Method', inplace=True)

# Find ranking and make
rank = []
for i, d in enumerate(datasets[:-2]):
    rank.append(np.array([df.loc[m, f"{d} MacroF1"] for m in methods]))
rank = np.array(rank).mean(axis=0)
tmp = np.array([[x, y]for y, x in sorted(zip(rank, methods))])
methods = list(tmp[:, 0])
rank = list(tmp[:, 1])

# make plot for each dataset
fig, axs = plt.subplots(2, 2, figsize=(23/2, 8), dpi=500)
width = 0.29
yticks = [i for i in np.arange(0.0, 1.01, 0.1)]
hatches = ["x", "O"]
colors = ["tab:purple", "tab:cyan"]
colors_hatch = ["indigo", "darkcyan"]
for i, d in enumerate(datasets):
    if i in [len(datasets)-1, len(datasets)-2]:
        metrics = {
            "MacroF1": np.array([df.loc[m, f"{d} MacroF1"] for m in methods if m != "Concerto"]),
            "Acc": np.array([df.loc[m, f"{d} Acc"] for m in methods if m != "Concerto"])
        }
        if i == 2:
            methods.remove("Concerto")
    else:
        metrics = {
            "MacroF1": np.array([df.loc[m, f"{d} MacroF1"] for m in methods]),
            "Acc": np.array([df.loc[m, f"{d} Acc"] for m in methods])
        }
    
    if i == 0:
        ax = axs[0][0]
        d = "PBMC CITE-seq (RNA + Protein)"
    elif i == 1:
        ax = axs[1][0]
        d = "BMMC (GEX + ADT)"
    elif i == 2:
        ax = axs[0][1]
        d = "PBMC CITE-seq (RNA)"
    elif i == 3:
        ax = axs[1][1]
        d = "BMMC (GEX)"

    multiplier = 0
    x = np.arange(len(methods)) - width/2
    
    for attribute, measurement in metrics.items():
        offset = width * multiplier
        rects = ax.bar(x+offset, measurement, width-0.015, label=attribute, hatch=hatches[multiplier], color=colors[multiplier], edgecolor=(colors_hatch[multiplier], 0.5))
        multiplier += 1

    
    # Function Mercator transform
    def forward(a):
        return a**3


    def inverse(a):
        a =  a**(1/3)
        a = np.nan_to_num(a)
        return a
    
    if i == 0 or i == 2:
        ax.set_yscale('function', functions=(forward, inverse))
    
    ax.grid(True, "major", axis="y", ls="--", linewidth=0.4, alpha=0.8)
    ax.set_title(d, fontsize=20,)  
    ax.margins(x=width/12)

    ax.set_ylim([0, 1])
    ax.set_yticks(yticks)
    if not(i == 0 or i == 1):
        ax.tick_params(axis="y", length=0.0)

    ax.set_xticks([n for n in range(len(methods))])
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=16,)
    ax.tick_params(axis="x", pad=-0.06)

    if i == 3:
        ax.set_xlabel("Methods", fontsize=20, loc='left')  
        # print(ax.xaxis.get_label().get_position())
        ax.xaxis.set_label_coords(-0.12, -0.45)

    if i == 1:
        ax.set_ylabel("Metrics", fontsize=20,) 
        ax.set_yticklabels(sum([["{:3.1f}".format(i), ""] for i in np.arange(0.0, 1.0, 0.2)], []) + ["1.0"], fontsize=17) # love it, I wanted a one liner :)
    elif i == 0:
        ax.set_yticklabels(["0.0", "", "", "", "", "", "0.6", "", "0.8", "0.9", "1.0"], fontsize=17) # love it, I wanted a one liner :)
    else:
        ax.set_yticklabels([""] * len(yticks))

handles, labels = axs[0][1].get_legend_handles_labels()

labels[0] = "Macro F1"
labels[1] = "Accuracy"
plt.legend(handles, labels,
           loc="upper center",
           ncol=3,
           fontsize=20,
           bbox_to_anchor=[0.0, 3.28]
           )  

plt.subplots_adjust(left=0.042, right=0.99, top=0.86, bottom=0.152, wspace=0.02, hspace=0.7)
plt.savefig("plots/tableB3_f.svg")
