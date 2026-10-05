# MRI → SPARC SDS dataset workflow

Converts breast MRI DICOM series into a SPARC-compliant (SDS 2.0.0) dataset, with
placeholders for the segmentation masks that get added later.

Workflows differ **only** in how a case's DICOM folders are located; everything
after that is identical.

| workflow | how folders are located | contrasts produced |
|---|---|---|
| `volunteer` | a path template, e.g. `{root}/{case}/MRI_T2_prone` | 1 (single-phase) |
| `adhb` | a manifest json listing series folders per case | 5 (one dynamic series split by acquisition number) |
| `ea` | scans an EA1141-style tree, identifying series from DICOM headers | 2 (T1 pre + T1 post, separate folders) |

**Using your own data?** You do not have to fit one of these. Start with
`--discover` (below) and work from the manifest it writes, or describe your layout
with `--template`. No code changes needed either way.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env     # then edit .env
```

Data locations and case identifiers live in `.env`, which is gitignored — **nothing
that identifies real data belongs in a tracked file.** There are no hard-coded
fallbacks: an unset variable produces an error telling you which one to set.

| Variable | Used by | Meaning |
|---|---|---|
| `DICOM_ROOT_VOLUNTEER` | volunteer | substituted for `{root}` in the template |
| `SERIES_TEMPLATE_VOLUNTEER` | volunteer | path template, default `{root}/{case}/MRI_T2_prone` |
| `CASES_VOLUNTEER` | volunteer | comma-separated case ids |
| `MANIFEST_PATH_ADHB` | adhb | path to the manifest json |
| `CASES_ADHB` | adhb | comma-separated; empty = every case in the manifest |
| `DICOM_ROOT_EA` | ea | root holding one folder per subject |
| `CASES_EA` | ea | comma-separated subject ids |

Anything in `.env` can be overridden per run on the command line.

---

## Starting from an unfamiliar dataset

If you do not know how your DICOM is organised, let the code tell you. This is
**read-only — it converts nothing and modifies nothing**:

```bash
python main.py --discover /path/to/dicom_root
```

It finds every folder that directly contains `.dcm` files and identifies each one
from its DICOM header (`Modality`, `StudyDate`, `SeriesNumber`,
`SeriesDescription`) rather than its name, because folder naming is rarely
consistent across a dataset:

```
SUBJECT-0001
  [20190314] MG/MR
         3  Ax T2 ASPIR                        2 files
         4  Axial T1 FS pre                    3 files
         6  Axial T1 FS post                   3 files
         7  L CC Tomosynthesis                 1 files   (not MR, excluded)

Wrote 2 entries to data/discovered_manifest.json
```

The manifest it writes is immediately usable:

```json
{
  "SUBJECT-0001": {
    "contrast_dirs": ["/path/4.000000-Axial T1 FS pre",
                      "/path/6.000000-Axial T1 FS post"],
    "t2_dir": "/path/3.000000-Ax T2 ASPIR",
    "_all_mr_series": [ ... every MR series that was found ... ]
  }
}
```

`contrast_dirs` is a **guess** (T1 pre then post when the descriptions say so,
otherwise every non-T2 MR series). Check the order — the first entry becomes
`contrast_pre` — and prune what you do not want. Everything found is kept under
`_all_mr_series`, so a wrong guess never hides a series from you. Keys starting
with `_` are ignored by the loader.

Then point `MANIFEST_PATH_ADHB` at it and run:

```bash
python main.py --workflow adhb --cases SUBJECT-0001
```

`data/example_manifest.json` documents the format if you would rather write one by
hand.

---

## Quick start

### With Claude Code

The repo ships a skill at `.claude/skills/building-sds-datasets/`. Claude loads it
automatically — just describe the run in plain language:

> Switch to volunteer and run cases 00001, 00002 and 00003. Put the masks from
> `/path/to/masks` into each case's `model_predicted_LPS_nii` and
> `researcher_manual_LPS_nii`.

The skill tells Claude to build the right command, to stop and ask before anything
destructive, and — importantly — to **verify the result with md5 comparisons**
rather than trusting the exit code. Expect it to report which case mapped to which
`sub-N`, because that mapping is not recorded anywhere in the dataset.

Nothing in the skill is magic: it is a short Markdown file documenting the same
CLI described below. Read it if you want to know exactly what Claude will do.

### Without Claude

Run `main.py` directly. The same request as above:

```bash
python main.py --workflow volunteer \
               --cases 00001 00002 00003 \
               --masks /path/to/masks
```

That is the whole thing — **you never need to edit the source to configure a run.**
The constants in `main.py` only read `.env`; every one of them has a flag.

---

## CLI reference

```
python main.py [--discover DIR] [--discover-out FILE]
               [--workflow {volunteer,adhb,ea}] [--cases ...]
               [--template TPL]... [--t2-template TPL] [--root DIR]
               [--masks DIR] [--mask-types ...]
               [--out-root DIR] [--name NAME] [--force]
               [--with-t2 | --no-t2]
```

| Flag | Default | Meaning |
|---|---|---|
| `--discover DIR` | — | read-only: report the layout under DIR, write a manifest, exit |
| `--discover-out FILE` | `data/discovered_manifest.json` | where `--discover` writes |
| `--workflow` | `WORKFLOW` in `main.py` | which way to locate folders |
| `--cases` | the workflow's list from `.env` | case / subject ids to process |
| `--template TPL` | — | path template with `{root}` / `{case}`; repeat for several series |
| `--t2-template TPL` | — | template for the T2 series |
| `--root DIR` | the workflow's root from `.env` | value substituted for `{root}` |
| `--masks DIR` | none | folder of existing masks to inject |
| `--mask-types` | `model_predicted_LPS_nii researcher_manual_LPS_nii` | which sample types the masks fill |
| `--out-root DIR` | `generated_datasets/` | where datasets land |
| `--name NAME` | `{workflow}_{YYYYmmdd-HHMMSS}` | dataset folder name |
| `--force` | off | delete and rebuild if the name already exists |
| `--with-t2` / `--no-t2` | per workflow | override T2 conversion |

### Describing a layout with `--template`

When you can write the path down, you do not need a manifest. `{root}` and
`{case}` are substituted per case:

```bash
# one series per case
python main.py --template "{root}/{case}/MRI_T2_prone" \
               --root /path/to/dicom --cases CASE-01 CASE-02

# contrasts in separate folders -> repeat the flag; they map onto
# contrast_pre, contrast_1, ... in the order given
python main.py --template "{root}/{case}/T1_pre" \
               --template "{root}/{case}/T1_post" \
               --t2-template "{root}/{case}/T2" \
               --root /path/to/dicom --cases CASE-01
```

`--template` overrides whatever the chosen `--workflow` would otherwise do, so it
works with any of them.

`python main.py --help` always shows the current list.

### Examples

```bash
# Three volunteer cases, with masks
python main.py --workflow volunteer --cases 00001 00002 00003 \
               --masks /path/to/masks

# One EA1141 subject -> two datasets, one per MRI timepoint, from one command
python main.py --workflow ea --cases SUBJECT-0001

# ADHB without T2, under a name you choose
python main.py --workflow adhb --cases CL00001 --no-t2 --name adhb_CL00001_noT2

# Also fill ground truth from the same mask folder
python main.py --workflow volunteer --cases 00001 --masks /path/to/masks \
               --mask-types model_predicted_LPS_nii researcher_manual_LPS_nii ground_truth_LPS_nii
```

### Mask injection

A mask file is matched to a case when **the case id appears in the filename**
(case-insensitive). So `VL00001_mask.nii.gz` matches case `00001`, and an
ADHB-style `CL00001_mask.nii.gz` matches `CL00001` — no prefix is hard-coded.

- No match → warning, the placeholder stays empty, other cases continue.
- Two or more matches → the run **aborts**. Tidy the folder or rename, rather than
  letting the wrong mask into the dataset.

---

## Output layout

Every run writes to `generated_datasets/<name>/`, where `<name>` is timestamped by
default, so **runs never overwrite one another**. If the name already exists and is
non-empty the run aborts *before* converting anything; `--force` deletes it instead.

```
generated_datasets/
  volunteer_20261005-115404/        <- the SDS dataset
    primary/sub-1/sam-1/...
    samples.xlsx  subjects.xlsx  manifest.xlsx  dataset_description.xlsx  ...
  _work/
    volunteer_20261005-115404/      <- intermediates, safe to delete afterwards
```

`ea` produces two datasets from a single command, `<name>_tp1` and `<name>_tp2`,
one per MRI timepoint.

`generated_datasets/` is in `.gitignore`.

---

## Sample type convention

Every file in the dataset carries a `sample_type`, which also determines its
filename: the segment after the final underscore is the extension, and the full
token is the stem (`nii` expands to `nii.gz`).

```
contrast_pre_nrrd        -> contrast_pre_nrrd.nrrd
model_predicted_LPS_nii  -> model_predicted_LPS_nii.nii.gz
findings_json            -> findings_json.json
```

| Origin | `sample_type` |
|---|---|
| Original DICOM, pre-contrast | `contrast_pre_nrrd` |
| Original DICOM, contrast phase N | `contrast_N_nrrd` |
| T2 series | `t2_nrrd` |
| Registered image, pre-contrast | `registration_pre_nrrd` |
| Registered image, contrast phase N | `registration_N_nrrd` |
| Model-predicted segmentation | `model_predicted_LPS_nii` |
| Researcher manual segmentation | `researcher_manual_LPS_nii` |
| Ground truth segmentation | `ground_truth_LPS_nii` |
| Vessel mask | `vessel_mask_nii` |
| Findings metadata | `findings_json` |

Which types a run produces depends on the workflow:

| workflow | contrast | registration | T2 | manual |
|---|---|---|---|---|
| `volunteer` | `_pre` | `_pre` | — | all 5 |
| `adhb` | `_pre`, `_1` … `_4` | `_pre`, `_1` … `_4` | `t2_nrrd` | all 5 |
| `ea` | `_pre`, `_1` | `_pre`, `_1` | `t2_nrrd` | all 5 |

`registration_*` starts out as a byte copy of the matching `contrast_*`; the real
registration overwrites it later.

---

## Verifying a run

The exit code is not evidence. Check all three:

1. **Log** — one `[mask] case <id>: <file> -> [...]` line per case, and no
   `No mask for case` or `failed:` lines.
2. **Bytes** — each injected mask must be byte-identical to its source, and the
   case ↔ `sub-N` mapping must not be crossed:

   ```bash
   md5sum /path/to/masks/*.nii.gz
   find generated_datasets/<name>/primary -name "*_LPS_nii.nii.gz" -exec md5sum {} +
   ```

   A 0-byte file means injection failed.
3. **`samples.xlsx`** — the expected `sample type` for every sam, one set per
   subject, and the subject count equal to the number of cases requested.

---

## Gotchas

- **`sub-N` ordering follows `--cases` order, and the case id is not stored in the
  dataset.** Record the mapping yourself.
- **An empty `.nrrd` is dropped from the dataset; empty `.nii` / `.json`
  placeholders are kept.** So a failed contrast conversion makes that `sam-` folder
  vanish silently — check `samples.xlsx`, not just the exit code.
- **Redirecting output to a file on Windows** may need `PYTHONIOENCODING=utf-8`.
- **Windows 260-character path limit** — keep working directories shallow; a deep
  `--out-root` fails with a confusing `FileNotFoundError` on a path that was just
  created.
- **Sources live on the `Z:` drive.** `DICOM folder not found` usually means the
  share is not mounted, not that the case is missing.

---

## Pipeline internals

For each case:

1. Resolve its DICOM folders (`build_volunteer_sources` / `build_adhb_sources` /
   `build_ea_sources`). `ea` reads the DICOM `Modality` header to tell MRI from
   mammography and `StudyDate` to order the visits, because EA1141's folder names
   are not consistent enough to rely on.
2. Create one placeholder folder per `sample_type`.
3. Inject masks over the matching placeholders (`--masks`).
4. Convert each series folder to NRRD in parallel (`dcmfolder2nrrd`). A folder
   containing subfolders is already split by contrast; a folder of loose dcm files
   is split by Acquisition Number. Contrasts from several folders are concatenated
   in order, then mapped onto `contrast_types`.
5. Copy each contrast into its `registration_*` placeholder.
6. Arrange the sample folders into `sam-N` and add them to the dataset via
   `sparc_me`'s `Subject()` / `Sample()`, setting each `sample type`.

Each converted NRRD carries its intensity `min` / `max` in the header, so
downstream tools (registration, front-end window/level) do not have to rescan the
volume.

---

## Requirements

```bash
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` with your own data locations (see [Setup](#setup)). `.env` is
gitignored; keep real paths and case identifiers out of tracked files.
