# %%

import json
import matplotlib.pyplot as plt
import statistics
import os
import utils

os.makedirs("generated/", exist_ok=True)
plt.rcParams["font.family"] = "serif"
LANGS = ["de", "ru", "es"]

plt.figure(figsize=(3.5, 2.5))
ax = plt.gca()

with open("ranking/metric_track1/track1_score_dict.json", "r") as f:
    data = json.load(f)

data = [
    {
        "name": sys,
        # es doesn't have term acc
        "x": statistics.mean([
            data[lang]["proper"][sys]["proper_term_success_rate"]
            for lang in LANGS
            if data[lang]["proper"][sys]["proper_term_success_rate"] > 0.3
        ]),
        "y": statistics.mean([
            data[lang]["proper"][sys]["chrf2++"]
            for lang in LANGS
            if data[lang]["proper"][sys]["chrf2++"] > 20
        ]),
    }
    for sys in [
        sys
        for sys, val in data["de"]["proper"].items()
        if val != {} and "chrf2++" in data["de"]["proper"][sys]
    ]
]
data = [
    line for line in data
    if line["y"] > 55
]
plt.scatter(
    [x["x"] for x in data],
    [x["y"] for x in data],
    color='#999',
    marker='.',
    s=120,
    zorder=-100,
)

PARETO = {
    "1": [line for line in data if line["name"] in {"o3-term-guide", "laniqo"}],
    "2": [line for line in data if line["name"] in {"duterm"}],
    "3": [line for line in data if line["name"] in {"Erlendur", "MeGuMa"}],
}
plt.scatter(
    [x["x"] for x in PARETO["1"]],
    [x["y"] for x in PARETO["1"]],
    color='#f3ce12',
    marker=r'$\star$',
    s=100,
    zorder=-100,
)
plt.scatter(
    [x["x"] for x in PARETO["2"]],
    [x["y"] for x in PARETO["2"]],
    color='#d39927',
    marker=r'$\star$',
    s=100,
    zorder=-100,
)
plt.scatter(
    [x["x"] for x in PARETO["3"]],
    [x["y"] for x in PARETO["3"]],
    color='#d37527',
    marker=r'$\star$',
    s=100,
    zorder=-100,
)
for line in data:
    plt.text(
        line["x"],
        line["y"],
        utils.SYS_TO_NAME_2.get(line["name"], line["name"]),
        fontsize=7,
        ha='center' if line["name"] != "salamandrata" else 'right',
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
plt.savefig("generated/tradeoff_accuracy.pdf")

# %%

plt.figure(figsize=(3.5, 2.5))
ax = plt.gca()

with open("ranking/metric_track1/track1_score_dict.json", "r") as f:
    data = json.load(f)
data = [
    {
        "name": sys,
        # es doesn't have term acc
        "x": statistics.mean([
            data[lang]["proper"][sys]["consistency_frequent"]
            for lang in LANGS
            if data[lang]["proper"][sys]["consistency_frequent"]
        ]),
        "y": statistics.mean([
            data[lang]["proper"][sys]["chrf2++"]
            for lang in LANGS
            if data[lang]["proper"][sys]["chrf2++"] > 20
        ]),
    }
    for sys in [
        sys
        for sys, val in data["de"]["proper"].items()
        if val != {} and all([
            "chrf2++" in data[lang]["proper"][sys]
            for lang in LANGS
        ])
    ]
]
data = [
    line for line in data
    if line["y"] > 55
]

PARETO = {
    "1": [line for line in data if line["name"] in {"o3-term-guide", "laniqo"}],
    "2": [line for line in data if line["name"] in {"duterm"}],
    "3": [line for line in data if line["name"] in {"Erlendur", "MeGuMa"}],
}
pareto_all = {line["name"] for line in PARETO["1"] + PARETO["2"] + PARETO["3"]}
plt.scatter(
    [x["x"] for x in data if x["name"] not in pareto_all],
    [x["y"] for x in data if x["name"] not in pareto_all],
    color='#999',
    marker='.',
    s=120,
    zorder=-100,
)

plt.scatter(
    [x["x"] for x in PARETO["1"]],
    [x["y"] for x in PARETO["1"]],
    color='#f3ce12',
    marker=r'$\star$',
    s=100,
    zorder=-100,
)
plt.scatter(
    [x["x"] for x in PARETO["2"]],
    [x["y"] for x in PARETO["2"]],
    color='#d39927',
    marker=r'$\star$',
    s=100,
    zorder=-100,
)
plt.scatter(
    [x["x"] for x in PARETO["3"]],
    [x["y"] for x in PARETO["3"]],
    color='#d37527',
    marker=r'$\star$',
    s=100,
    zorder=-100,
)
for line in data:
    plt.text(
        line["x"],
        line["y"],
        utils.SYS_TO_NAME_2.get(line["name"], line["name"]),
        fontsize=7,
        ha='center' if line["name"] != "salamandrata" else 'right',
        va='center',
        zorder=-100,
    )

# plt.ylim(59.5, 72)
# plt.xlim(0.56, 1.0)
plt.yticks([60, 62, 64, 66, 68, 70, 72])

# set xticks formatter to percentages
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))

ax.spines[["top", "right"]].set_visible(False)
plt.xlabel('Terminology Consistency')
plt.ylabel('ChrF++')

plt.tight_layout(pad=0)
plt.savefig("generated/tradeoff_consistency.pdf")