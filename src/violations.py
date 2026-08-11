"""Map raw detections to structured PPE compliance findings."""

from __future__ import annotations

from typing import Any

# Official Construction-PPE class names (Ultralytics order)
CLASS_NAMES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "none",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
]

VIOLATION_CLASSES = {
    "no_helmet": "missing_helmet",
    "no_goggle": "missing_goggles",
    "no_gloves": "missing_gloves",
    "no_boots": "missing_boots",
}

WORN_CLASSES = {"helmet", "gloves", "vest", "boots", "goggles"}

# Association rules: for each person, require these worn classes nearby.
# Small/rarely-visible gear (gloves/boots/goggles) stays detector-only.
ASSOCIATION_REQUIRED = {
    "helmet": {
        "code": "missing_helmet",
        "severity": "high",
        "y_frac_max": 0.55,  # expect helmet in upper half of person box
    },
    "vest": {
        "code": "missing_vest",
        "severity": "medium",
        "y_frac_max": 0.85,
    },
}


def _normalize_class(name: str) -> str:
    return name.strip()


def _resolve_class(det: dict[str, Any]) -> str | None:
    if "class_name" in det:
        return _normalize_class(str(det["class_name"]))
    if "class_id" in det:
        cid = int(det["class_id"])
        return CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else None
    return None


def _center(bbox: list[float]) -> tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def _ppe_associated_with_person(
    person_bbox: list[float],
    ppe_bbox: list[float],
    y_frac_max: float,
) -> bool:
    """True if PPE center lies inside person box (optionally upper band)."""
    px1, py1, px2, py2 = [float(x) for x in person_bbox]
    if px2 <= px1 or py2 <= py1:
        return False
    cx, cy = _center(ppe_bbox)
    if not (px1 <= cx <= px2 and py1 <= cy <= py2):
        return False
    y_frac = (cy - py1) / (py2 - py1)
    return y_frac <= y_frac_max


def extract_violations(
    detections: list[dict[str, Any]],
    conf_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Convert detector outputs into business-level PPE violations.

    Each detection dict should contain at least:
      - class_name (str) or class_id (int)
      - confidence (float)
      - bbox (optional list [x1,y1,x2,y2])
    """
    findings: list[dict[str, Any]] = []
    for det in detections:
        conf = float(det.get("confidence", 0.0))
        if conf < conf_threshold:
            continue

        cls = _resolve_class(det)
        if cls is None:
            continue

        if cls in VIOLATION_CLASSES:
            findings.append(
                {
                    "code": VIOLATION_CLASSES[cls],
                    "class_name": cls,
                    "confidence": conf,
                    "bbox": det.get("bbox"),
                    "severity": "high" if cls == "no_helmet" else "medium",
                    "source": "detector",
                }
            )
    return findings


def infer_association_violations(
    detections: list[dict[str, Any]],
    conf_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Flag missing helmet/vest when a Person is present but the worn PPE
    is not associated to that person box. Complements weak no_* detectors.
    """
    kept = [d for d in detections if float(d.get("confidence", 0.0)) >= conf_threshold]
    persons = []
    by_class: dict[str, list[dict[str, Any]]] = {}
    for d in kept:
        cls = _resolve_class(d)
        if cls is None:
            continue
        by_class.setdefault(cls, []).append(d)
        if cls == "Person" and d.get("bbox") and len(d["bbox"]) == 4:
            persons.append(d)

    if not persons:
        return []

    # Explicit no_* already covers some people; still run association for worn gaps.
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[float, ...]]] = set()

    for person in persons:
        pb = [float(x) for x in person["bbox"]]
        pkey = tuple(round(x, 1) for x in pb)
        pconf = float(person.get("confidence", 0.0))

        for worn_name, meta in ASSOCIATION_REQUIRED.items():
            # Skip if an explicit matching no_* already fires on this person region
            explicit_cls = {
                "helmet": "no_helmet",
                "vest": None,  # dataset has no no_vest class
            }.get(worn_name)
            has_explicit = False
            if explicit_cls:
                for det in by_class.get(explicit_cls, []):
                    bb = det.get("bbox")
                    if not bb or len(bb) != 4:
                        continue
                    if _ppe_associated_with_person(pb, bb, y_frac_max=0.7):
                        has_explicit = True
                        break
            if has_explicit:
                continue

            associated = False
            for det in by_class.get(worn_name, []):
                bb = det.get("bbox")
                if not bb or len(bb) != 4:
                    continue
                if _ppe_associated_with_person(pb, bb, y_frac_max=float(meta["y_frac_max"])):
                    associated = True
                    break

            if associated:
                continue

            key = (meta["code"], pkey)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                {
                    "code": meta["code"],
                    "class_name": f"inferred_{worn_name}",
                    "confidence": pconf,
                    "bbox": pb,
                    "severity": meta["severity"],
                    "source": "association",
                }
            )

    return findings


def summarize_compliance(
    detections: list[dict[str, Any]],
    conf_threshold: float = 0.25,
    use_association: bool = True,
) -> dict[str, Any]:
    """Return a compliance summary for an image."""
    kept = [d for d in detections if float(d.get("confidence", 0.0)) >= conf_threshold]
    violations = extract_violations(kept, conf_threshold=conf_threshold)
    if use_association:
        # Association already skips people covered by an explicit no_* box.
        existing_codes_boxes = {
            (v["code"], tuple(round(float(x), 0) for x in (v.get("bbox") or [0, 0, 0, 0])))
            for v in violations
        }
        for v in infer_association_violations(kept, conf_threshold=conf_threshold):
            key = (
                v["code"],
                tuple(round(float(x), 0) for x in (v.get("bbox") or [0, 0, 0, 0])),
            )
            if key in existing_codes_boxes:
                continue
            violations.append(v)
            existing_codes_boxes.add(key)

    persons = sum(1 for d in kept if _resolve_class(d) == "Person")
    worn = sorted(
        {
            cls
            for d in kept
            if (cls := _resolve_class(d)) is not None and cls in WORN_CLASSES
        }
    )
    codes = sorted({v["code"] for v in violations})

    return {
        "compliant": len(violations) == 0,
        "person_count": persons,
        "worn_ppe": worn,
        "violation_codes": codes,
        "violations": violations,
        "detection_count": len(kept),
    }
