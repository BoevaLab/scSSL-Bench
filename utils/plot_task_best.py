import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale

def read_data(path: str):
    return pd.read_csv(path, sep=",")

# BC
df = read_data(path='results/table1.csv')
datasets = list([" ".join(col.split(" ")[:-1]) for col in df.columns if col != "Method" and col.endswith("Total")])
methods = list(df["Method"])
df.set_index('Method', inplace=True)

# Find ranking and make
rank_bc = []
for i, d in enumerate(datasets):
    rank_bc.append(np.array([df.loc[m, f"{d} Total"] for m in methods]))
rank_bc = np.array(rank_bc).mean(axis=0)
tmp = np.array([[x, y]for y, x in zip(rank_bc, methods)])
methods_bc = list(tmp[:, 0])
rank_bc = list(tmp[:, 1])


#  QR
df = read_data(path='results/tableB1_B2.csv')
datasets = list([" ".join(col.split(" ")[:-1]) for col in df.columns if col != "Method" and col.endswith("MacroF1")])
df.set_index('Method', inplace=True)

# Find ranking and make
rank_qr = []
for i, d in enumerate(datasets[:-1]):
    rank_qr.append(np.array([df.loc[m, f"{d} MacroF1"] for m in methods]))

df = read_data(path='results/table_B3.csv')
datasets = list([" ".join(col.split(" ")[:-1]) for col in df.columns if col != "Method" and col.endswith("MacroF1")])
df.set_index('Method', inplace=True)

# Find ranking and make
for i, d in enumerate(datasets[:-2]):
    rank_qr.append(np.array([df.loc[m, f"{d} MacroF1"] for m in methods]))
    
rank_qr = np.array(rank_qr).mean(axis=0)
tmp = np.array([[x, y]for y, x in zip(rank_qr, methods)])
methods_qr = list(tmp[:, 0])
rank_qr = list(tmp[:, 1])


# MP
df = read_data(path='results/table_B4.csv')
datasets = list([" ".join(col.split(" ")[:-1]) for col in df.columns if col != "Method"])
df.set_index('Method', inplace=True)

# Find ranking and make
rank_mp = []
for i, d in enumerate(datasets):
    rank_mp.append(np.array([df.loc[m, f"{d} Pearson"] for m in methods]))
rank_mp = np.array(rank_mp).mean(axis=0)
tmp = np.array([[x, y]for y, x in zip(rank_mp, methods)])
methods_mp = list(tmp[:, 0])
rank_mp = list(tmp[:, 1])


final_rates = np.array([rank_bc, rank_qr, rank_mp], dtype=float)
# final_rates = minmax_scale(final_rates, axis=1, )
print(final_rates)
print(final_rates.round(4))

rates_mean = final_rates.mean(axis=0)
methods = np.array([[x, y] for y, x in sorted(zip(rates_mean, methods), reverse=True)]) [:, 0]
final_rates = np.array([final_rates[:, y] for x, y in sorted(zip(rates_mean, range(final_rates.shape[1])), reverse=True)]).T.round(2)
print(methods)
print(final_rates.round(4))

methods_bc = list(tmp[:, 0])
rank_bc = list(tmp[:, 1])

min_val, max_val = 0.0, 1.0
n = 1000
orig_cmap = mpl.colormaps["Reds"].resampled(200)
colors = orig_cmap(np.linspace(min_val, max_val, n))
cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", colors)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(23/3.6, 4), dpi=500)
im = ax.imshow(final_rates, cmap=cmap, extent=[0,20,0,4], clim=[0.0, 1.0]) # jet tab20

xtick = [20/16, (20/16) * 3, (20/16) * 5, (20/16) * 7, (20/16) * 9, (20/16) * 11, (20/16) * 13, (20/16) * 15]
ytick = [(4/6) * 5, (4/6) * 3, (4/6)]
ax.set_xticks(xtick, labels=methods, fontsize=16,)
ax.set_yticks(ytick, labels=["Batch\nCorrection", "Query-to-Reference\nMapping", "Missing Modality\nPrediction"], fontsize=16)
ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=40, ha="right",
            rotation_mode="anchor", fontsize=16,)

# Loop over data dimensions and create text annotations.
for i in range(3):
    for j in range(8):
        # xtick[j], ytick[i]
        text = ax.text(xtick[j], ytick[i], final_rates[i, j],
                       ha="center", va="center", color="black")

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax, ticks=[0.0, 1.0])
cbar.ax.set_ylabel("Average Score", rotation=-90, va="bottom", fontsize=16, labelpad=-10)
cbar.ax.tick_params(labelsize=16)

ax.set_xlabel("Methods", fontsize=20, loc='center') 
ax.set_ylabel("Tasks", fontsize=20,) 
ax.set_title("Methods Comparison", fontsize=20)

ax.set_aspect(aspect="2.5")
# ax.set_aspect("equal")

plt.subplots_adjust(left=0.41, right=0.97, top=0.95, bottom=0.27, wspace=0.02, hspace=0.7)
plt.savefig("plots/table_comparison_reds.pdf")
plt.savefig("plots/table_comparison_reds.svg")
