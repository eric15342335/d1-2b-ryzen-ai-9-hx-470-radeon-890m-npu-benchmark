"""Compare each backend's decisions with the float32 PyTorch reference at matching packing.

Packing changes the long input's reference answers, so ONNX and Tomoali (all questions in one
forward) are compared with the packed reference and llama.cpp (one question per forward) with the
set_size=1 reference. Six decisions per backend: a numerical sanity check, not an accuracy benchmark.
"""

from typing import Any

from d1bench.common import WORKLOADS, Benchmark, load_request, read_json


def probabilities(answer: dict[str, Any]) -> dict[str, float]:
    """Probability of every label of one typed answer."""
    if answer["type"] == "noul":
        return {"yes": answer["noul"], "no": 1 - answer["noul"]}
    values: dict[str, float] = answer["probabilities"]
    return values


def label(answer: dict[str, Any], *, mean_score: bool = False) -> object:
    """Return the categorical decision; for Tomoali's mean ordinal score, derive the median level."""
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5  # noqa: PLR2004
    if answer["type"] == "choice":
        return answer["choice"]
    if not mean_score:
        return answer["score"]
    cumulative = 0.0
    for level, probability in sorted(answer["probabilities"].items(), key=lambda item: int(item[0])):
        cumulative += probability
        if cumulative >= 0.5:  # noqa: PLR2004
            return int(level)
    msg = f"score probabilities sum to {cumulative}"
    raise ValueError(msg)


def backend_answers(bench: Benchmark) -> list[tuple[str, dict[str, Any]]]:
    """(state, answers) per workload, in WORKLOADS order."""
    if bench.kind == "llama":
        return [
            (load_request(name)["state"], read_json(f"{bench.result}-{name}-output.json")["answers"])
            for name in WORKLOADS
        ]
    rows = read_json(bench.result)["workloads"]
    return [(row["state"], row["output"] if bench.kind == "npu" else row["output"]["answers"]) for row in rows]


def evaluate(benchmarks: list[Benchmark], reference: Benchmark) -> list[dict[str, Any]]:
    """Per-question label agreement and largest probability difference for every backend."""
    refs = {(w["state"], w["set_size"]): w["output"]["answers"] for w in read_json(reference.result)["workloads"]}
    records = []
    for bench in benchmarks:
        if bench.kind not in {"onnx", "npu", "llama"}:
            continue
        mean_score = bench.kind == "npu"
        set_size = 1 if bench.kind == "llama" else None
        for index, (state, answers) in enumerate(backend_answers(bench)):
            expected = refs[state, set_size]
            comparisons = []
            for question, actual in answers.items():
                got, want = probabilities(actual), probabilities(expected[question])
                comparisons.append(
                    {
                        "question": question,
                        "label": label(actual, mean_score=mean_score),
                        "reference_label": label(expected[question]),
                        "label_agrees": label(actual, mean_score=mean_score) == label(expected[question]),
                        "max_abs_probability_delta": max(abs(got[k] - want[k]) for k in want),
                        "raw_score": actual.get("score"),
                    },
                )
            records.append(
                {
                    "id": bench.id,
                    "backend": bench.label,
                    "workload": WORKLOADS[index],
                    "reference_set_size": set_size,
                    "score_semantics": "mean; median derived for the label" if mean_score else "median",
                    "comparisons": comparisons,
                },
            )
    return records
