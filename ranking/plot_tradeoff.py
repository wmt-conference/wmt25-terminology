# %%

import json
import matplotlib.pyplot as plt
import statistics
import os

os.makedirs("../generated/", exist_ok=True)
plt.rcParams["font.family"] = "serif"

plt.figure(figsize=(3, 2.5))
ax = plt.gca()

LANGS = ["de", "ru", "es"]
with open("metric_track1/scores/track1_score_dict.json", "r") as f:
    data = json.load(f)
data = [
    {
        "name": sys,
        # es doesn't have term acc
        "x": statistics.mean([data[lang]["random"][sys]["term_success_rate"] for lang in LANGS]),
        "y": statistics.mean([data[lang]["random"][sys]["chrf2++"] for lang in LANGS]),
    }
    for sys in [
        sys
        for sys, val in data["de"]["proper"].items()
        if val != {}
    ]
]
data = [
    line for line in data
    if line["y"] > 55
]

plt.scatter(
    [x["x"] for x in data],
    [x["y"] for x in data],
    color='#44c',
    marker='x',
    s=30
)
for line in data:
    plt.text(
        line["x"],
        line["y"]-0.5,
        line["name"],
        fontsize=8,
        ha='center',
        va='top'
    )

plt.ylim(58, 70)
plt.xlim(0.6, 1.0)

# set xticks formatter to percentages
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))

ax.spines[["top", "right"]].set_visible(False)
plt.xlabel('Terminology Accuracy')
plt.ylabel('ChrF++')

plt.tight_layout()
# save in PDF to be lossless
plt.savefig("../generated/tradeoff.pdf")

# TODO:
# manually add pareto spline

# %%


with open("metric_track1/scores/track1_score_dict.json", "r") as f:
    data = json.load(f)

systems = {
    lang: set(val["proper"].keys())
    for lang, val in data.items()
}

# %%
