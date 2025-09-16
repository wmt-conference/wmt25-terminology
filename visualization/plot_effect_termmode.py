# %%

import json
import matplotlib.pyplot as plt
import statistics
import os
import utils

os.makedirs("generated/", exist_ok=True)
plt.rcParams["font.family"] = "serif"

plt.figure(figsize=(3.5, 4))
ax = plt.gca()

LANGS = ["de", "ru", "es"]
with open("ranking/metric_track1/track1_score_dict.json", "r") as f:
    data = json.load(f)
data = [
    {
        "name": sys,
        # es doesn't have term acc
        "y": [
            statistics.mean([data[lang]["noterm"][sys]["chrf2++"] for lang in LANGS]),
            statistics.mean([data[lang]["random"][sys]["chrf2++"] for lang in LANGS]),
            statistics.mean([data[lang]["proper"][sys]["chrf2++"] for lang in LANGS]),
        ]
    }
    for sys in [
        sys
        for sys, val in data["de"]["proper"].items()
        if val != {}
    ]
]
data = [
    line for line in data
    if statistics.mean(line["y"]) > 55
]
data.sort(
    key=lambda line: statistics.mean(line["y"]),
    reverse=True,
)

for line_i, line in enumerate(data):
    plt.plot(
        [line_i],
        [line["y"][0]],
        color="black",
        marker="x",
    )
    # white "background" for R
    plt.plot(
        [line_i],
        [line["y"][1]],
        color="white",
        marker="s",
        markersize=5,
        zorder=-5,
    )
    plt.plot(
        [line_i+0.08],
        [line["y"][1]],
        color="black",
        marker="$R$",
    )
    plt.plot(
        [line_i],
        [line["y"][2]],
        color="black",
        marker=r"$\star$",
    )
    plt.plot(
        [line_i]*3,
        line["y"],
        color="black",
        zorder=-10,
    )

plt.ylim(54, 72)

# set xticks formatter to percentages
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))

ax.spines[["top", "right"]].set_visible(False)
# plt.xlabel('Terminology Accuracy')
plt.xticks(
    range(len(data)),
    [utils.SYS_TO_NAME_2.get(line["name"], line["name"]) for line in data],
    rotation=90,
    fontsize=8,
)
plt.ylabel('ChrF++')

plt.tight_layout(pad=0)
plt.savefig("generated/effect_termmode.pdf")