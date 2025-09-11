# %%

import json
import matplotlib.pyplot as plt
import statistics
import os
import utils

os.makedirs("../generated/", exist_ok=True)
plt.rcParams["font.family"] = "serif"

plt.figure(figsize=(3.5, 2.5))
ax = plt.gca()

LANGS = ["de", "ru", "es"]
with open("metric_track1/scores/track1_score_dict.json", "r") as f:
    data = json.load(f)
data = [
    {
        "name": sys,
        # es doesn't have term acc
        "x": statistics.mean([data[lang]["proper"][sys]["term_success_rate"] for lang in LANGS]),
        "y": statistics.mean([data[lang]["proper"][sys]["chrf2++"] for lang in LANGS]),
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
    color='black',
    marker='.',
    s=50
)
for line in data:
    plt.text(
        line["x"],
        line["y"],
        utils.SYS_TO_NAME_2.get(line["name"], line["name"]),
        fontsize=7,
        ha='center',
        va='center',
        zorder=-100,
    )

plt.ylim(59.5, 72)
plt.xlim(0.56, 1.0)
plt.yticks([60, 62, 64, 66, 68, 70, 72])

# set xticks formatter to percentages
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))

ax.spines[["top", "right"]].set_visible(False)
plt.xlabel('Terminology Accuracy')
plt.ylabel('ChrF++')

plt.tight_layout(pad=0)
plt.savefig("../generated/tradeoff.pdf")