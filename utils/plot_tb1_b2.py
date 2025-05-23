import pandas as pd 

import matplotlib.pyplot as plt
import numpy as np

def read_data(path: str):
    return pd.read_csv(path, sep=",")

df = read_data(path='results/tableB1_B2.csv')

datasets = list([" ".join(col.split(" ")[:-1]) for col in df.columns if col != "Method" and col.endswith("MacroF1")])

methods = list(df["Method"])
df.set_index('Method', inplace=True)

# Find ranking and make
rank = []
for i, d in enumerate(datasets[:-1]):
    rank.append(np.array([df.loc[m, f"{d} MacroF1"] for m in methods]))
rank = np.array(rank).mean(axis=0)
tmp = np.array([[x, y]for y, x in sorted(zip(rank, methods))])
methods = list(tmp[:, 0])
rank = list(tmp[:, 1])

# make plot for each dataset
fig, axs = plt.subplots(1, 5, figsize=(23, 6), dpi=500)

width = 0.29
yticks = [i for i in np.arange(0.0, 1.01, 0.1)]
hatches = ["x", "O"]
colors = ["tab:purple", "tab:cyan"]
colors_hatch = ["indigo", "darkcyan"]
for i, d in enumerate(datasets):
    if i == len(datasets)-1:
        metrics = {
            "MacroF1": np.array([df.loc[m, f"{d} MacroF1"] for m in methods if m != "Concerto"]),
            "Acc": np.array([df.loc[m, f"{d} Acc"] for m in methods if m != "Concerto"])
        }
        methods = methods[1:]
    else:
        metrics = {
            "MacroF1": np.array([df.loc[m, f"{d} MacroF1"] for m in methods]),
            "Acc": np.array([df.loc[m, f"{d} Acc"] for m in methods])
        }
    

    multiplier = 0
    x = np.arange(len(methods)) - width/2
    
    for attribute, measurement in metrics.items():
        offset = width * multiplier
        rects = axs[i].bar(x+offset, measurement, width-0.015, label=attribute, hatch=hatches[multiplier], color=colors[multiplier], edgecolor=(colors_hatch[multiplier], 0.5))
        multiplier += 1
    
    axs[i].grid(True, "major", axis="y", ls="--", linewidth=0.4, alpha=0.8)
    axs[i].set_title(d, fontsize=20,)  
    axs[i].margins(x=width/12)

    axs[i].set_ylim([0, 1])
    axs[i].set_yticks(yticks)
    if i != 0:
        axs[i].tick_params(axis="y", length=0.0)

    axs[i].set_xticks([n for n in range(len(methods))])
    axs[i].set_xticklabels(methods, rotation=35, ha='right', fontsize=16,)
    axs[i].tick_params(axis="x", pad=-0.06)

    if i == 2:
        axs[i].set_xlabel("Methods", fontsize=20)  
        
    if i == 0:
        axs[i].set_ylabel("Metrics", fontsize=20,) 
        axs[i].set_yticklabels(sum([["{:3.1f}".format(i), ""] for i in np.arange(0.0, 1.0, 0.2)], []) + ["1.0"], fontsize=17) # love it, I wanted a one liner :)
    else:
        axs[i].set_yticklabels([""] * len(yticks))

handles, labels = axs[0].get_legend_handles_labels()

labels[0] = "Macro F1"
labels[1] = "Accuracy"
plt.legend(handles, labels,
           loc="lower center",
           ncol=3,
           fontsize=20,
           bbox_to_anchor=[-1.54, 1.12])  


plt.subplots_adjust(left=0.037, right=0.99, top=0.815, bottom=0.237, wspace=0.04)
plt.savefig("plots/tableB1_B2.svg")
