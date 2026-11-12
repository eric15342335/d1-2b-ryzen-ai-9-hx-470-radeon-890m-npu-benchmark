"""Compare the measured decision outputs with the pinned float32 reference."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8-sig"))


def probabilities(answer):
    values = ({"yes": answer["noul"], "no": 1 - answer["noul"]}
              if answer["type"] == "noul" else answer["probabilities"])
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in values.values())
    assert abs(sum(values.values()) - 1) < 1e-5
    return values


def label(answer, mean_score=False):
    if answer["type"] == "noul":
        return answer["noul"] >= .5
    if answer["type"] == "choice":
        return answer["choice"]
    if not mean_score:
        return answer["score"]
    cumulative = 0
    for level, probability in sorted(answer["probabilities"].items(), key=lambda x: int(x[0])):
        cumulative += probability
        if cumulative >= .5:
            return int(level)
    raise AssertionError("Invalid score distribution")


def main():
    reference = read("pytorch-reference.json")
    refs = {(w["state"], w["set_size"]): w["output"]["answers"] for w in reference["workloads"]}
    requests = [json.loads((ROOT / "fixtures" / (n + "-common.json")).read_text())
                for n in ("ticket", "long-ticket")]
    cases = []
    for backend, file, mean_score in [
        ("ONNX CPU q4", "cpu-q4-ticket-common-forward.json", False),
        ("ONNX WebGPU q4", "webgpu-q4-ticket-common-forward.json", False),
        ("Tomoali CPU+NPU W4", "tomoali-npu.json", True),
    ]:
        data = read(file)
        for index, row in enumerate(data["workloads"]):
            answers = row["output"] if mean_score else row["output"]["answers"]
            cases.append((backend, index, row["state"], None, answers, mean_score))
    for tag in ("q4km", "q8"):
        for index, name in enumerate(("ticket", "long-ticket")):
            output = read(f"llama-{tag}-{name}-output.json")
            assert output["usage"]["output_tokens"] == 0
            cases.append((f"llama.cpp {tag}", index, requests[index]["state"], 1,
                          output["answers"], False))
    records = []
    for backend, index, state, packing, answers, mean_score in cases:
        expected = refs[state, packing]
        assert answers.keys() == expected.keys()
        comparisons = []
        for question, actual in answers.items():
            actual_p, expected_p = probabilities(actual), probabilities(expected[question])
            assert actual_p.keys() == expected_p.keys()
            comparisons.append({
                "question": question, "label": label(actual, mean_score),
                "reference_label": label(expected[question]),
                "label_agrees": label(actual, mean_score) == label(expected[question]),
                "max_abs_probability_delta": max(abs(actual_p[k] - expected_p[k]) for k in actual_p),
                "raw_score": actual.get("score"),
            })
        records.append({"backend": backend, "workload_index": index,
                        "reference_set_size": packing,
                        "score_semantics": "mean; median derived from probabilities for label comparison"
                        if mean_score else "median",
                        "comparisons": comparisons})
    result = {"reference_model_revision": reference["model_revision"],
              "reference_dtype": reference["dtype"],
              "scope": "two text fixtures, three questions each; not a general accuracy evaluation",
              "records": records}
    (RESULTS / "correctness.json").write_text(json.dumps(result, indent=2) + "\n")
    for backend in dict.fromkeys(r["backend"] for r in records):
        checks = [c for r in records if r["backend"] == backend for c in r["comparisons"]]
        print(backend, f"{sum(c['label_agrees'] for c in checks)}/{len(checks)} labels",
              "max probability delta", round(max(c["max_abs_probability_delta"] for c in checks), 6))


if __name__ == "__main__":
    main()
