"""Unit tests for PPE violation mapping."""

from src.violations import (
    extract_violations,
    infer_association_violations,
    summarize_compliance,
)


def test_extract_missing_helmet():
    dets = [
        {"class_name": "Person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
        {"class_name": "no_helmet", "confidence": 0.8, "bbox": [1, 1, 5, 5]},
        {"class_name": "helmet", "confidence": 0.7, "bbox": [2, 2, 6, 6]},
    ]
    findings = extract_violations(dets, conf_threshold=0.25)
    assert len(findings) == 1
    assert findings[0]["code"] == "missing_helmet"
    assert findings[0]["severity"] == "high"


def test_conf_threshold_filters():
    dets = [{"class_name": "no_gloves", "confidence": 0.1}]
    assert extract_violations(dets, conf_threshold=0.25) == []


def test_summarize_compliant_when_no_missing():
    dets = [
        {"class_name": "Person", "confidence": 0.95},
        {"class_name": "helmet", "confidence": 0.9},
        {"class_name": "vest", "confidence": 0.88},
    ]
    summary = summarize_compliance(dets)
    assert summary["compliant"] is True
    assert summary["violation_codes"] == []
    assert "helmet" in summary["worn_ppe"]
    assert summary["person_count"] == 1


def test_summarize_noncompliant():
    dets = [
        {"class_name": "Person", "confidence": 0.9},
        {"class_name": "no_boots", "confidence": 0.7},
        {"class_name": "no_goggle", "confidence": 0.6},
    ]
    summary = summarize_compliance(dets)
    assert summary["compliant"] is False
    assert set(summary["violation_codes"]) == {"missing_boots", "missing_goggles"}


def test_class_id_resolution():
    # 7 = no_helmet in official Construction-PPE order
    dets = [{"class_id": 7, "confidence": 0.9}]
    findings = extract_violations(dets)
    assert findings[0]["code"] == "missing_helmet"


def test_association_flags_missing_helmet():
    # Person box with no helmet nearby → inferred violation
    dets = [
        {"class_name": "Person", "confidence": 0.95, "bbox": [0, 0, 100, 200]},
        {"class_name": "vest", "confidence": 0.9, "bbox": [20, 80, 80, 160]},
    ]
    findings = infer_association_violations(dets)
    codes = {f["code"] for f in findings}
    assert "missing_helmet" in codes
    assert all(f["source"] == "association" for f in findings)


def test_association_ok_when_helmet_on_person():
    dets = [
        {"class_name": "Person", "confidence": 0.95, "bbox": [0, 0, 100, 200]},
        {"class_name": "helmet", "confidence": 0.9, "bbox": [30, 10, 70, 50]},
        {"class_name": "vest", "confidence": 0.9, "bbox": [20, 80, 80, 160]},
    ]
    summary = summarize_compliance(dets)
    assert summary["compliant"] is True
    assert "missing_helmet" not in summary["violation_codes"]


def test_association_skips_when_explicit_no_helmet():
    dets = [
        {"class_name": "Person", "confidence": 0.95, "bbox": [0, 0, 100, 200]},
        {"class_name": "no_helmet", "confidence": 0.8, "bbox": [30, 10, 70, 50]},
        {"class_name": "vest", "confidence": 0.9, "bbox": [20, 80, 80, 160]},
    ]
    assoc = infer_association_violations(dets)
    assert not any(f["code"] == "missing_helmet" for f in assoc)
    summary = summarize_compliance(dets)
    assert summary["violation_codes"] == ["missing_helmet"]
    assert summary["violations"][0]["source"] == "detector"
