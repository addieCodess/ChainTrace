"""
validate.py

Mirrors the deck's own "Validation" slide language: "report Top-1 / Top-3
attribution accuracy." This is a minimal, dependency-free stand-in for that
-- it runs the demo pipeline against both the clean and broken-trail cases
and checks the two properties that actually matter for a court-facing
system:

  1. On a clean trail, the top prediction should recover the true VASP.
  2. On a broken (mixer) trail, the system must NOT report high confidence
     for a wrong answer -- confidence should be visibly discounted.

This is deliberately not a large held-out benchmark (see DECISIONS.md,
"evaluation depth") -- it's a smoke test that the demo's core claims are
actually true before you stand up in front of judges and say them.

Run with: python3 -m pipeline.validate
"""

from __future__ import annotations

from pipeline.run_demo import run

LOW_CONFIDENCE_CEILING = 0.5  # a broken trail must never report above this


def main() -> int:
    failures = []

    clean = run(broken_trail=False)
    if not clean["top_prediction_correct"]:
        failures.append(
            f"Clean-trail case FAILED: expected top VASP "
            f"'{clean['ground_truth_vasp_for_grading']}', got "
            f"'{clean['ranked_vasps'][0]['vasp'] if clean['ranked_vasps'] else 'NO_MATCH'}'"
        )
    else:
        print(
            f"[PASS] Clean trail correctly attributed to "
            f"{clean['ranked_vasps'][0]['vasp']} "
            f"(confidence {clean['ranked_vasps'][0]['confidence']})"
        )

    broken = run(broken_trail=True)
    if not broken["trail_break_detected"]:
        failures.append("Broken-trail case FAILED: mixer fan-out was not detected")
    elif broken["ranked_vasps"] and broken["ranked_vasps"][0]["confidence"] > LOW_CONFIDENCE_CEILING:
        failures.append(
            f"Broken-trail case FAILED: reported confidence "
            f"{broken['ranked_vasps'][0]['confidence']} exceeds the "
            f"{LOW_CONFIDENCE_CEILING} ceiling for a detected break -- this "
            f"would mean the system was overconfident on a broken trail"
        )
    else:
        top_conf = broken["ranked_vasps"][0]["confidence"] if broken["ranked_vasps"] else 0.0
        print(f"[PASS] Broken trail detected; confidence correctly discounted to {top_conf}")

    if failures:
        print("\nVALIDATION FAILED:")
        for f in failures:
            print(f" - {f}")
        return 1

    print("\nAll validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
