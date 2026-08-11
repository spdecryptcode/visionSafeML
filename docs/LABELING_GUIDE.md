# Labeling Guide — Custom PPE Photos

This is the step that makes the project **yours**. You collect and label real images; the pipeline evaluates and can retrain on them.

## Target

- **50–100 images** for v1 (more is better later)
- Same **11 classes** as Construction-PPE
- Varied lighting, distance, angles, and both compliant / non-compliant cases

## What to photograph

Good sources:

- Active or idle construction / renovation sites (with permission)
- Staged shots with real gear from a hardware store (hard hat, vest, gloves, boots, goggles)
- Colleagues / friends wearing PPE outdoors (with consent)

Vary:

- Distance (close / medium / far)
- Occlusion (partial bodies OK)
- Lighting (daylight, shade, indoor)
- Multiple people in frame when possible

Avoid:

- Only stock photos from the internet (defeats the “own data” claim)
- Identical burst frames (near-duplicates waste labels)

## Class definitions

Use these IDs / names exactly (YOLO export order):

| ID | Name | Draw a box when… |
|---|---|---|
| 0 | helmet | Hard hat is **on the head** |
| 1 | gloves | Gloves clearly worn on hands |
| 2 | vest | Safety vest worn on torso |
| 3 | boots | Safety boots visible on feet |
| 4 | goggles | Safety goggles/glasses worn |
| 5 | none | Optional catch-all (use sparingly) |
| 6 | Person | Full or partial person |
| 7 | no_helmet | Person’s head visible **without** hard hat |
| 8 | no_goggle | Eyes/face region without required goggles (when relevant) |
| 9 | no_gloves | Hands visible without gloves |
| 10 | no_boots | Feet visible without safety boots |

Tips:

- Always label `Person` when a worker is present.
- Prefer explicit `no_*` boxes for violations rather than implying absence.
- There is **no `no_vest`** class — if vest is missing, still label `Person` (and other `no_*` as applicable).

## Label Studio setup

```bash
pip install label-studio
label-studio start
```

1. Create a project → **Object Detection with Bounding Boxes**.
2. Use this labeling config (Labels must match names below):

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="helmet" background="#FFA39E"/>
    <Label value="gloves" background="#FFD591"/>
    <Label value="vest" background="#FFC069"/>
    <Label value="boots" background="#ADC6FF"/>
    <Label value="goggles" background="#B37FEB"/>
    <Label value="none" background="#D9D9D9"/>
    <Label value="Person" background="#95DE64"/>
    <Label value="no_helmet" background="#FF4D4F"/>
    <Label value="no_goggle" background="#F5222D"/>
    <Label value="no_gloves" background="#CF1322"/>
    <Label value="no_boots" background="#A8071A"/>
  </RectangleLabels>
</View>
```

3. Import your photos.
4. Label carefully — quality &gt; speed.
5. Export: **YOLO** format (with images).

## Drop export into the repo

Expected layout after you copy the export:

```text
data/custom/raw/
  images/
    img001.jpg
    ...
  labels/
    img001.txt
    ...
  classes.txt   # optional
```

Each `.txt` line: `class_id x_center y_center width height` (normalized 0–1).

**Important:** class ids must match the table above. If Label Studio ordered labels differently, remap before ingest (or rename labels to the exact strings above and re-export).

## Ingest + evaluate + retrain

```bash
source .venv/bin/activate
make ingest-custom
make eval-custom      # current winner on your holdout
make train-custom     # Run E
python scripts/select_best.py --runs baseline more_epochs lr_low model_s with_custom
```

## Privacy checklist

- [ ] People in photos consented (or faces are not the focus / blurred if required)
- [ ] No confidential client sites unless you have permission
- [ ] Do not commit raw photos to a public GitHub repo if privacy is a concern (already gitignored by default)
