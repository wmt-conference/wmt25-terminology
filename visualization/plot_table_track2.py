# %%
import os
import json
import statistics
import collections
import utils

os.makedirs("generated/", exist_ok=True)

LANGS = ["enzh", "zhen"]
with open("ranking/metric_track2/track2_score_dict.json", "r") as f:
    data = json.load(f)

systems = [
    sys
    for sys, val in data["enzh"]["proper"].items()
]

# compute zscores of variables for system ranking
data_agg = collections.defaultdict(list)
for lang, lang_v in data.items():
    for task, task_v in lang_v.items():
        for sys, sys_v in task_v.items():
            for metric, val in sys_v.items():
                if val == -1:
                    continue
                # print(val)
                data_agg[(lang, task, metric)].append(val)
data_agg = {
    k: (statistics.mean(v), statistics.stdev(v))
    for k, v in data_agg.items()
}
for lang, lang_v in data.items():
    for task, task_v in lang_v.items():
        for sys, sys_v in task_v.items():
            for metric, val in list(sys_v.items()):
                if val == -1:
                    continue
                sys_v[metric+"_z"] = (
                    (val - data_agg[(lang, task, metric)][0]) /
                    data_agg[(lang, task, metric)][1]
                )

systems.sort(
    key=lambda sys: statistics.mean(
        [data[lang]["proper"][sys]["chrf2++"]+data[lang]["proper"][sys]["lowercase_term_success_rate"] for lang in LANGS]),
    reverse=True,
)

# %%

def color_cell_chrf(val):
    color = f"SeaGreen3!{max(0, min(95, (val-40)*4.5)):.0f}!Firebrick3!50"
    return f"\\cellcolor{{{color}}} {val:.1f}"


def color_cell_acc(val):
    color = f"SeaGreen3!{max(0, min(95, (val-60)*3)):.0f}!Firebrick3!50"
    return f"\\cellcolor{{{color}}} {val:.1f}"


def color_cell_cons(val):
    color = f"SeaGreen3!{max(0, min(95, (val-80)*10)):.0f}!Firebrick3!50"
    return f"\\cellcolor{{{color}}} {val:.1f}"

def nocolor_cell(val):
    return f"{val:.1f}"

with open("generated/track2.tex", "w") as f:
    print(
        r"\begin{tabular}{l  cvv cvv cvv |c    cvv cvv |c     cvv}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\bf Proper, ChrF} & \multicolumn{3}{c}{\bf Proper, Acc.} & \multicolumn{3}{c|}{\bf Proper, Cons.} & & \multicolumn{3}{c}{\bf Random, ChrF} & \multicolumn{3}{c|}{\bf Random, Acc.} & & \multicolumn{3}{c}{\bf NoTerm, ChrF}\\",
        r"\bf System  & \bf Avg & \bf EnZh & \bf ZhEn   & \bf Avg & \bf EnZh & \bf ZhEn   & \bf Avg & \bf EnZh & \bf ZhEn  & & \bf Avg & \bf EnZh & \bf ZhEn   & \bf Avg & \bf EnZh & \bf ZhEn  & & \bf Avg & \bf EnZh & \bf ZhEn \\",
        r"\midrule",
        sep="\n",
        file=f,
    )

    for sys in systems:
        print(
            utils.SYS_TO_NAME.get(sys, sys),
            # proper, chrf
            color_cell_chrf(statistics.mean([
                data[lang]["proper"][sys]["chrf2++"] for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["proper"][sys]["chrf2++"])
                for lang in LANGS
            ],
            # proper, term
            color_cell_acc(statistics.mean([
                data[lang]["proper"][sys]["lowercase_term_success_rate"]*100 for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["proper"][sys]["lowercase_term_success_rate"]*100)
                for lang in LANGS
            ],
            # proper, cons
            color_cell_acc(statistics.mean([
                data[lang]["proper"][sys]["consistency_frequent"]*100 for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["proper"][sys]["consistency_frequent"]*100)
                for lang in LANGS
            ],
            "",
            # random, chrf
            color_cell_chrf(statistics.mean([
                data[lang]["random"][sys]["chrf2++"] for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["random"][sys]["chrf2++"])
                for lang in LANGS
            ],
            # random, term
            color_cell_acc(statistics.mean([
                data[lang]["random"][sys]["lowercase_term_success_rate"]*100 for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["random"][sys]["lowercase_term_success_rate"]*100)
                for lang in LANGS
            ],
            "",
            # noterm, chrf
            color_cell_chrf(statistics.mean([
                data[lang]["noterm"][sys]["chrf2++"] for lang in LANGS
            ])),
            *[
                nocolor_cell(data[lang]["noterm"][sys]["chrf2++"])
                for lang in LANGS
            ],
            sep=" & ",
            end="\\\\\n",
            file=f,
        )

    print(
        r"\bottomrule",
        r"\end{tabular}",
        sep="\n",
        file=f,
    )
