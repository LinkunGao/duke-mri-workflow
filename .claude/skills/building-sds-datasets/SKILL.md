---
name: building-sds-datasets
description: Use when the user asks to build, generate or re-run a SPARC SDS dataset in this repo - mentions a workflow (volunteer / adhb / ea), names case or subject ids to run, asks what DICOM is available or how an unfamiliar dataset is organised, asks to put existing masks into model_predicted_LPS_nii or researcher_manual_LPS_nii, or refers to the generated_datasets output folders.
---

# Building SDS datasets

`main.py` converts DICOM into a SPARC SDS dataset. Everything is driven by CLI
flags and `.env` — **never edit constants in `main.py` to configure a run.** The
constants only read `.env`; editing them leaves uncommitted noise and makes the
run unreproducible.

## The command

```bash
python main.py --workflow volunteer --cases <ids...> --masks <dir>
```

| Flag | Meaning |
|---|---|
| `--workflow {volunteer,adhb,ea}` | how to locate each case's folders |
| `--cases ...` | case / subject ids; omit to use `.env` |
| `--template TPL` | path template with `{root}` / `{case}`; repeat for several series |
| `--root DIR` | value for `{root}` |
| `--masks DIR` | folder of existing masks; omit to leave placeholders empty |
| `--mask-types ...` | default `model_predicted_LPS_nii researcher_manual_LPS_nii` |
| `--with-t2` / `--no-t2` | override the workflow's T2 default |
| `--name NAME` | dataset folder name; default `{workflow}_{YYYYmmdd-HHMMSS}` |
| `--force` | allow overwriting an existing name (deletes it) |

Run `python main.py --help` for the current list.

## Configuration lives in .env

Data roots and case ids come from `.env` (gitignored). An unset variable produces
an actionable error naming it. **Never write a real path or case id into a tracked
file** — not into `main.py`, the README, or this skill. If the user gives you one,
it goes in `.env` or on the command line.

## Unfamiliar dataset? Discover first

When the user asks what data is there, or their layout does not match a workflow,
run the read-only inspection rather than guessing:

```bash
python main.py --discover <dicom root>
```

It converts nothing. It prints every series with its Modality / StudyDate /
SeriesNumber / SeriesDescription and writes a manifest. **The `contrast_dirs` it
writes is a guess** — show the user the discovered series and confirm the order
(first entry becomes `contrast_pre`) before converting. Everything found is kept
under `_all_mr_series`.

Then either point `MANIFEST_PATH_ADHB` at the manifest and use `--workflow adhb`,
or, if the paths are regular, use `--template`.

## Mask matching

A file is matched to a case when the **filename contains the case id**
(case-insensitive). No match → warning, placeholder stays empty. Two matches → the
run aborts; ask the user which one rather than guessing.

## Output

Every dataset lands in `generated_datasets/<name>/`, where `<name>` defaults to
`{workflow}_{timestamp}` — **each run is a new folder; nothing is overwritten.**
`ea` produces two, `<name>_tp1` and `<name>_tp2`, one per MRI timepoint, from a
single command. Intermediates go to `generated_datasets/_work/<name>/`.

If the name already exists and is non-empty the run aborts before converting
anything. Only pass `--force` (which deletes that folder) when the user has said
to discard it.

## Verify before reporting success

Never claim the dataset is correct from the log alone. Check all three:

1. `[mask] case <id>: <file> -> [...]` appears once per case, and no
   `No mask for case` or `failed:` lines.
2. Each injected file is **byte-identical to its source** — compare md5 of
   `generated_datasets/<name>/primary/sub-N/sam-M/<type>.nii.gz` against the
   source mask, and confirm case ↔ sub-N did not get crossed. A 0-byte file means
   injection failed.
3. `samples.xlsx` lists the expected `sample type` for every sam, one set per
   subject, and the subject count equals the number of cases requested.

Report what the checks showed, not that you ran them.

## Gotchas

- `sub-N` ordering follows `--cases` order; the case id is **not** recorded in
  the dataset. Keep the mapping in your report.
- Empty `.nrrd` is dropped from the dataset; empty `.nii`/`.json` placeholders
  are kept. So a sample type with no data silently has no `sam-` folder.
- Redirecting output to a file may need `PYTHONIOENCODING=utf-8` on Windows.
- Windows 260-char path limit: keep working directories shallow.
- A "folder not found" error often means the network share is not mounted, not
  that the case is missing.
