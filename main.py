import sparc_me as sm
from sparc_me import Dataset, Sample, Subject
from file_utils import delete_folder, is_empty_file
from tools import dcmfolder2nrrd, first_dicom_header
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import argparse
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

dataset = None

# Anchor every relative path to this file's directory, so results are identical
# no matter what the current working directory is
SCRIPT_DIR = Path(__file__).resolve().parent

# Real data locations and case identifiers are read from .env, which is
# gitignored, so none of them end up in the repository. Copy .env.example to .env
# and fill it in. There are deliberately NO hard-coded fallbacks for the data
# roots: an unset variable raises a clear error instead of silently pointing
# somewhere wrong.
load_dotenv(SCRIPT_DIR / ".env")


def _env(name, default=None):
    """Read an environment variable, treating blank as unset."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_path(name, default=None):
    value = _env(name)
    return Path(value) if value else default


def _env_list(name, default=None):
    """Read a comma-separated environment variable into a list."""
    value = _env(name)
    if not value:
        return default if default is not None else []
    return [item.strip() for item in value.split(",") if item.strip()]


def require(value, env_name, what):
    """Fail with an actionable message when a required .env value is missing."""
    if value in (None, "", []):
        raise SystemExit(
            f"{env_name} is not set.\n"
            f"Copy .env.example to .env and set {env_name} to {what}.")
    return value

# ---------------------------------------------------------------------------
# Switch: "volunteer" / "adhb" / "ea"
#   volunteer: DICOM path = dicom_root/{case}/MRI_T2_prone; single-phase series,
#              producing 1 contrast
#   adhb:      DICOM path = the t1 / t2 path for each case in
#              data/ADHB_dicom_latest.json; dynamic series, producing several
#              contrasts
#   ea:        DICOM path = the EA1141 public dataset. Each subject holds several
#              exams (MRI + mammography); the MRI ones are picked out by the
#              DICOM Modality header and ordered into 1st / 2nd visit by
#              StudyDate. T1 pre and post are two separate series folders, each
#              producing 1 contrast.
# Every case hands dcmfolder2nrrd one or more series root folders
# (contrast_dirs), and each folder decides for itself how to split contrasts:
#   folder contains subfolders -> already split; one nrrd per subfolder
#                                 (loose dcm files in the root are ignored)
#   folder contains only dcm   -> group by acquisition number; as many nrrds as
#                                 there are contrasts mixed together
# Contrasts from several folders are concatenated in contrast_dirs order and then
# filled into contrast_types in sequence.
# The rest of the pipeline (convert to nrrd -> copy for registration -> arrange
# into sam-N -> add to dataset) is identical for all three.
# ---------------------------------------------------------------------------
WORKFLOW = "ea"

# Whether the generated nrrd files are compressed.
#   False (default): uncompressed nrrd -- larger files, but faster read/write and
#                    no decompression needed downstream
#   True:            gzip compressed nrrd
NRRD_USE_COMPRESSION = False


# TODO 1: create dataset
def createWorkflowDataset(dest_dir, title):
    global dataset
    delete_folder(dest_dir)
    # Subject / Sample numbering uses class-level counters in sparc_me, which are
    # not reset when a second dataset is created in the same process (ea hits this
    # by building two timepoints in one run, making the second dataset start at
    # sub-2). Reset on every dataset creation so each one starts at sub-1 / sam-1.
    Subject.count = 0
    Sample.count = 0
    dataset = sm.Dataset()
    # NOTE: Step2, way1: load dataset from template
    dataset.set_path(dest_dir)
    dataset.create_empty_dataset(version='2.0.0')
    # manifest = dataset.get_metadata(metadata_file="manifest")
    # manifest.data['patient_id'] = "nan"

    dataset.save()
    _update_dataset_description(title)


# Helper: map a sample_type token to its file name. The trailing segment after the
# last underscore is the extension; the full token is kept as the file stem.
#   contrast_pre_nrrd -> contrast_pre_nrrd.nrrd
#   model_predicted_LPS_nii -> model_predicted_LPS_nii.nii.gz
def sample_type_to_filename(sample_type):
    fmt = sample_type.rsplit("_", 1)[1]
    if fmt == "nii":
        fmt = "nii.gz"
    return f"{sample_type}.{fmt}"


# TODO 1.5: resolve each case's DICOM folders (the only difference between workflows)
def build_template_sources(templates, case_numbers, root=None, t2_template=None):
    """Locate each case's series folders from path templates.

    A template is an ordinary string with two placeholders, {root} and {case},
    e.g. "{root}/{case}/MRI_T2_prone". Give several templates when the contrasts
    live in separate folders (EA-style T1 pre + T1 post); they map onto
    contrast_types in the order they are listed.

    This covers any layout where the path can be written down. When it cannot,
    use --discover to inspect the actual folders and write a manifest instead.

    :param templates: one or more path templates
    :type templates: list[str]
    :param case_numbers: case ids to resolve
    :type case_numbers: list[str]
    :param root: value substituted for {root}
    :type root: Path | str | None
    :param t2_template: optional template for the T2 series
    :type t2_template: str | None
    :return: [{"name": case, "contrast_dirs": [Path], "t2_dir": Path | None}]
    """
    root = str(root) if root is not None else ""
    sources = []
    for case in case_numbers:
        contrast_dirs = []
        for template in templates:
            dicom_dir = Path(template.format(root=root, case=case))
            if not dicom_dir.is_dir():
                print(f"DICOM folder not found, skip case {case}: {dicom_dir}")
                contrast_dirs = None
                break
            contrast_dirs.append(dicom_dir)
        if not contrast_dirs:
            continue

        t2_dir = None
        if t2_template:
            candidate = Path(t2_template.format(root=root, case=case))
            if candidate.is_dir():
                t2_dir = candidate
            else:
                print(f"T2 folder not found for case {case}, leaving t2 empty: {candidate}")

        sources.append({"name": case, "contrast_dirs": contrast_dirs, "t2_dir": t2_dir})
    return sources


def build_manifest_sources(metadata_path, case_numbers=None, with_t2=True):
    """Read each case's series folders from a manifest json.

    Two shapes are accepted. The generic one, which --discover writes and which
    new users should use::

      {"SUBJECT-0001": {"contrast_dirs": ["/path/T1 pre", "/path/T1 post"],
                        "t2_dir": "/path/T2"},
       "SUBJECT-0002": {}}

    and the legacy ADHB one::

      {"CL00001": {"latest": {"t1": {"path": ..., "Orientation Tag": "LPS"},
                              "t2": {"path": ..., "Orientation Tag": "LPS"}}}}

    Every path is a series root folder; pointing at one of the dcm files inside
    it also works, since the parent directory is used. Each folder is handed to
    dcmfolder2nrrd, which decides whether to split contrasts by subfolder or by
    acquisition number. Contrasts from several folders are concatenated in order
    and mapped onto contrast_pre / contrast_1 ...

    Keys starting with "_" are ignored, so --discover can leave notes in the file.
    Cases with an empty entry ({}) have no usable DICOM and are skipped.

    :param metadata_path: path of the manifest json
    :type metadata_path: Path
    :param case_numbers: run only this subset of cases; empty means every case in
                         the manifest
    :type case_numbers: list[str] | None
    :param with_t2: False = do not convert T2 (the t2_* samples stay empty)
    :type with_t2: bool
    :return: [{"name": case, "contrast_dirs": [Path], "t2_dir": Path | None}]
    """
    if not Path(metadata_path).is_file():
        raise SystemExit(
            f"Manifest not found: {metadata_path}\n"
            f"Point MANIFEST_PATH_ADHB in .env at your manifest, or generate one "
            f"with:  python main.py --discover <dicom root>\n"
            f"See data/example_manifest.json for the format.")
    metadata = _load_json(metadata_path)

    cases = case_numbers if case_numbers else [
        key for key in metadata if not key.startswith("_")]
    sources = []
    for case in cases:
        entry = metadata.get(case) or {}
        if not entry:
            print(f"No DICOM path in manifest, skip case {case}")
            continue

        if "contrast_dirs" in entry:
            contrast_dirs = _existing_series_dirs(case, entry["contrast_dirs"])
            if contrast_dirs is None:
                continue
            t2_dir = None
            if with_t2 and entry.get("t2_dir"):
                t2_dirs = _existing_series_dirs(case, [entry["t2_dir"]])
                t2_dir = t2_dirs[0] if t2_dirs else None
        else:
            # legacy ADHB shape
            latest = entry.get("latest")
            if not latest:
                print(f"No DICOM path in manifest, skip case {case}")
                continue
            t1_dir = _adhb_series_dir(case, latest, "t1")
            if t1_dir is None:
                continue
            contrast_dirs = [t1_dir]
            t2_dir = _adhb_series_dir(case, latest, "t2") if with_t2 else None

        sources.append({
            "name": case,
            "contrast_dirs": contrast_dirs,
            "t2_dir": t2_dir,
        })
    return sources


def _existing_series_dirs(case, raw_paths):
    """Resolve a list of manifest paths to existing series folders.
    Returns None when any of them is missing, so the case is skipped as a whole
    rather than silently losing one contrast."""
    resolved = []
    for raw in raw_paths:
        series_dir = _series_dir_of(raw)
        if not series_dir.is_dir():
            print(f"DICOM folder not found, skip case {case}: {series_dir}")
            return None
        resolved.append(series_dir)
    return resolved


def build_ea_sources(ea_root, subject_ids, timepoint, with_t2=False):
    """ea: the EA1141 public dataset, laid out as

      EA1141/{subject}/{exam folder}/{series folder}/*.dcm

    A subject holds several exams with MRI and mammography mixed together, and the
    naming follows no pattern at all (MRI may be called "MR BREAST RESEARCH EXAM"
    / "MRIBB3" / "ECOG-ACRIN", mammography "MMSCRCOMBO" / "Standard Screening -
    Combo"), so the DICOM Modality header decides, keeping only MR.

    The remaining MRI exams are sorted by study date ascending and the exam at
    index `timepoint` is taken (0 = first, 1 = second). Do not read this as
    "year": a subject's two exams can fall in the same calendar year, and some
    subjects have only one exam, in which case they are skipped in the second
    dataset.

    Within the chosen exam only the two T1 phases are taken (SeriesDescription
    containing t1 plus pre / post), mapping to contrast_pre / contrast_1; all
    other derived series (MIP, DISCO, ...) are ignored.

    :param ea_root: EA1141 root directory (subject folders sit directly inside)
    :type ea_root: Path
    :param subject_ids: subject folder names to process; None means every folder
                        under ea_root
    :type subject_ids: list[str] | None
    :param timepoint: which MRI exam, 0 = first, 1 = second
    :type timepoint: int
    :param with_t2: False (default) = do not convert T2, the t2_* samples stay
                    empty; True = also convert the exam's T2 series into t2_nrrd
    :type with_t2: bool
    :return: [{"name": subject, "contrast_dirs": [pre_dir, post_dir], "t2_dir": Path | None}]
    """
    ea_root = Path(ea_root)
    subjects = subject_ids if subject_ids else sorted(
        p.name for p in ea_root.iterdir() if p.is_dir())

    sources = []
    for subject in subjects:
        subject_dir = ea_root / subject
        if not subject_dir.is_dir():
            print(f"Subject folder not found, skip {subject}: {subject_dir}")
            continue

        mri_exams = _ea_mri_exams(subject, subject_dir)
        if len(mri_exams) <= timepoint:
            print(f"Only {len(mri_exams)} MRI exam(s), skip {subject} "
                  f"at timepoint {timepoint + 1}")
            continue

        exam_dir = mri_exams[timepoint][1]
        contrast_dirs = _ea_t1_series_dirs(subject, exam_dir)
        if contrast_dirs is None:
            continue

        t2_dir = _ea_t2_series_dir(subject, exam_dir) if with_t2 else None
        print(f"[ea] {subject} timepoint {timepoint + 1}: {exam_dir.name} -> "
              f"{[p.name for p in contrast_dirs]}"
              f"{f' + T2 {t2_dir.name}' if t2_dir else ''}")
        sources.append({
            "name": subject,
            "contrast_dirs": contrast_dirs,
            "t2_dir": t2_dir,
        })
    return sources


def _ea_mri_exams(subject, subject_dir):
    """All exam folders under a subject with Modality == MR, sorted by study date
    ascending.

    :return: [(date_key, exam_dir)], where date_key is "YYYYMMDD"
    :rtype: list[tuple[str, Path]]
    """
    exams = []
    for exam_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        header = first_dicom_header(exam_dir, ["Modality", "StudyDate"])
        if header is None:
            print(f"No readable DICOM, skip exam {subject}/{exam_dir.name}")
            continue
        if str(header.get("Modality") or "").upper() != "MR":
            continue
        exams.append((_ea_study_date(exam_dir.name, header.get("StudyDate")), exam_dir))
    exams.sort(key=lambda item: item[0])
    return exams


def _ea_study_date(exam_name, header_date):
    """Study date: prefer the StudyDate header, otherwise parse the MM-DD-YYYY
    prefix of the folder name.

    When neither is available, return "99999999" so the exam sorts last and is
    therefore never mistaken for the first visit.
    """
    if header_date:
        return str(header_date)
    match = re.match(r"(\d{2})-(\d{2})-(\d{4})", exam_name)
    if match:
        month, day, year = match.groups()
        return f"{year}{month}{day}"
    print(f"No study date for exam {exam_name}, sorted last")
    return "99999999"


def _ea_t1_series_dirs(subject, exam_dir):
    """The exam's T1 pre / post series folders. Returns None if either is missing.

    The decision is based on SeriesDescription (falling back to the folder name):
    containing t1 and pre -> pre, containing t1 and post -> post. When several
    match the same phase, the lowest series number wins.
    """
    candidates = {"pre": [], "post": []}
    for series_dir in sorted(p for p in exam_dir.iterdir() if p.is_dir()):
        header = first_dicom_header(series_dir, ["SeriesDescription"])
        desc = str((header or {}).get("SeriesDescription") or series_dir.name).lower()
        if "t1" not in desc:
            continue
        for phase in ("pre", "post"):
            if phase in desc:
                candidates[phase].append((_ea_series_number(series_dir.name), series_dir))

    resolved = []
    for phase in ("pre", "post"):
        if not candidates[phase]:
            print(f"No T1 {phase} series, skip {subject}/{exam_dir.name}")
            return None
        if len(candidates[phase]) > 1:
            print(f"{subject}/{exam_dir.name}: {len(candidates[phase])} T1 {phase} series, "
                  f"taking the lowest series number {sorted(candidates[phase])[0][1].name}")
        resolved.append(sorted(candidates[phase])[0][1])
    return resolved


def _ea_t2_series_dir(subject, exam_dir):
    """The exam's T2 series folder (SeriesDescription containing t2), or None if
    there is none. When several match, the lowest series number wins. Only reached
    when EA_WITH_T2 = True."""
    candidates = []
    for series_dir in sorted(p for p in exam_dir.iterdir() if p.is_dir()):
        header = first_dicom_header(series_dir, ["SeriesDescription"])
        desc = str((header or {}).get("SeriesDescription") or series_dir.name).lower()
        if "t2" in desc:
            candidates.append((_ea_series_number(series_dir.name), series_dir))
    if not candidates:
        print(f"No T2 series in {subject}/{exam_dir.name}, leaving t2 empty")
        return None
    if len(candidates) > 1:
        print(f"{subject}/{exam_dir.name}: {len(candidates)} T2 series, "
              f"taking the lowest series number {sorted(candidates)[0][1].name}")
    return sorted(candidates)[0][1]


def _ea_series_number(series_name):
    """The series number prefixing a series folder name, e.g.
    "4.000000-Axial T1 FS pre Asset-81676" -> 4.0. Unparseable names sort last."""
    match = re.match(r"([\d.]+)", series_name)
    return float(match.group(1)) if match else float("inf")


def discover_dicom_layout(root, out_path):
    """Read-only: report how DICOM is actually laid out under `root` and write a
    manifest you can edit and feed straight back in.

    Nothing is converted and nothing under `root` is modified.

    A "series folder" is any folder that directly contains .dcm files, so this
    works regardless of how deeply they are nested. Each one is identified from
    its DICOM header (Modality / StudyDate / SeriesNumber / SeriesDescription)
    rather than its name, because folder naming is rarely consistent across a
    dataset. Non-MR series (mammography and the like) are reported but left out
    of the manifest.

    :param root: directory to scan
    :type root: Path | str
    :param out_path: where to write the manifest json
    :type out_path: Path
    :return: the manifest that was written
    :rtype: dict
    """
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"--discover: not a directory: {root}")

    print(f"Scanning {root} ...", flush=True)
    found = {}
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        dcm_count = sum(1 for _ in folder.glob("*.dcm"))
        if not dcm_count:
            continue
        header = first_dicom_header(
            folder, ["Modality", "StudyDate", "SeriesNumber", "SeriesDescription"])
        if header is None:
            print(f"  unreadable DICOM, skipped: {folder}")
            continue
        relative = folder.relative_to(root)
        subject = relative.parts[0] if relative.parts else root.name
        found.setdefault(subject, []).append({
            "modality": str(header.get("Modality") or "?").upper(),
            "study_date": str(header.get("StudyDate") or ""),
            "series": header.get("SeriesNumber"),
            "description": str(header.get("SeriesDescription") or folder.name),
            "files": dcm_count,
            "path": str(folder),
        })

    if not found:
        raise SystemExit(f"--discover: no DICOM found under {root}")

    manifest = {
        "_note": "Generated by --discover. Edit contrast_dirs / t2_dir as needed, "
                 "then run: python main.py --workflow adhb --cases <key> "
                 "(with MANIFEST_PATH_ADHB pointing here). Keys starting with "
                 "_ are ignored.",
    }
    for subject, series_list in sorted(found.items()):
        print(f"\n{subject}")
        studies = {}
        for item in sorted(series_list, key=lambda s: (s["study_date"], _series_sort(s))):
            studies.setdefault(item["study_date"], []).append(item)
        for study_date, items in sorted(studies.items()):
            modalities = sorted({item["modality"] for item in items})
            print(f"  [{study_date or 'no date'}] {'/'.join(modalities)}")
            for item in items:
                flag = "" if item["modality"] == "MR" else "   (not MR, excluded)"
                number = item["series"] if item["series"] is not None else "?"
                print(f"      {str(number):>4}  {item['description'][:40]:<40} "
                      f"{item['files']:>5} files{flag}")

            mr_items = [item for item in items if item["modality"] == "MR"]
            if not mr_items:
                continue
            key = subject if len(studies) == 1 else f"{subject}__{study_date}"
            manifest[key] = _suggest_manifest_entry(mr_items)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    entries = len([k for k in manifest if not k.startswith("_")])
    print(f"\nWrote {entries} entr{'y' if entries == 1 else 'ies'} to {out_path}")
    print("Review it (especially contrast_dirs order: pre first, then post) "
          "before running a conversion.")
    return manifest


def _suggest_manifest_entry(mr_items):
    """Guess contrast_dirs / t2_dir for one study's MR series.

    T1 pre / post are preferred when their descriptions say so; otherwise every
    non-T2 MR series is listed in series-number order and left for a human to
    prune. The full list is kept under _all_mr_series either way, so nothing that
    was found gets hidden by a wrong guess.
    """
    def described(item):
        return item["description"].lower()

    t2 = [item for item in mr_items if "t2" in described(item)]
    non_t2 = [item for item in mr_items if item not in t2]

    pre = [item for item in non_t2 if "pre" in described(item)]
    post = [item for item in non_t2 if "post" in described(item)]
    if pre and post:
        contrast = [pre[0], post[0]]
    else:
        contrast = non_t2

    entry = {"contrast_dirs": [item["path"] for item in contrast]}
    if t2:
        entry["t2_dir"] = t2[0]["path"]
    entry["_all_mr_series"] = [
        {"series": item["series"], "description": item["description"],
         "files": item["files"], "path": item["path"]}
        for item in mr_items
    ]
    return entry


def _series_sort(item):
    """Sort series by series number, pushing missing numbers to the end."""
    number = item.get("series")
    try:
        return float(number)
    except (TypeError, ValueError):
        return float("inf")


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _adhb_series_dir(case, entry, key):
    """The series folder behind entry["latest"][key]; prints the reason and
    returns None when it cannot be resolved."""
    node = entry.get(key) or {}
    if not node.get("path"):
        print(f"No {key} path in metadata, skip {key} of case {case}")
        return None
    series_dir = _series_dir_of(node["path"])
    if not series_dir.is_dir():
        print(f"{key} DICOM folder not found, skip {key} of case {case}: {series_dir}")
        return None
    return series_dir


def _series_dir_of(raw_path):
    """A path in the json may be the series folder itself or one of the dcm files
    inside it."""
    path = Path(raw_path)
    return path if path.is_dir() else path.parent


# TODO 2: build the SDS dataset from the resolved DICOM folders (single automatic pass)
def generate_self_data_structure(case_sources, self_dest_dir, config,
                                 use_compression=NRRD_USE_COMPRESSION,
                                 mask_dir=None, mask_types=None):
    """
    Build the SDS dataset end-to-end for each case:
      1. generate the per-sample-type placeholder folder structure
      2. convert the contrast (and optionally T2) DICOM series to nrrd (parallel)
      3. distribute the converted nrrds into their sample_type folders, and copy
         each contrast once more for registration
      4. reorganize sample_type folders into sam-N
      5. move samples into the dataset and set the sample type

    :param case_sources: [{"name", "contrast_dirs", "t2_dir"}], produced by
                         build_*_sources
    :type case_sources: list[dict]
    :param self_dest_dir: temp working directory
    :type self_dest_dir: Path
    :param config: one entry of WORKFLOW_CONFIGS, holding sample_types /
                   contrast_types / registration_types / t2_types
    :type config: dict
    :param use_compression: whether the generated nrrds are compressed; defaults
                            to NRRD_USE_COMPRESSION (uncompressed)
    :type use_compression: bool
    :param mask_dir: folder holding existing masks (--masks). None = no injection,
                     placeholders stay empty
    :type mask_dir: Path | None
    :param mask_types: which sample_types the masks fill, default DEFAULT_MASK_TYPES
    :type mask_types: list[str] | None
    """
    sample_types = config["sample_types"]
    contrast_types = config["contrast_types"]
    registration_types = config["registration_types"]
    t2_types = config["t2_types"]

    all_temp_cases = []
    case_dirs = []

    # ---- pass 1: placeholder structure + collect conversion jobs ----
    for source in case_sources:
        case = source["name"]

        samples_dir = self_dest_dir / case / "samples"
        processed_dir = self_dest_dir / case / "processed"
        # processed holds pure intermediates and is regenerated on every run. It
        # MUST be cleared first: move_files sweeps *every* sam-N in this directory
        # into the dataset, so without clearing it, sam folders left over from a
        # previous run get written in as if they were this run's result (especially
        # dangerous when this run's conversion fails). If the previous run was a
        # different workflow, a leftover sam-N beyond this run's sample_types
        # length also raises IndexError outright.
        delete_folder(processed_dir)
        samples_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        #     TODO 3.1 generate the per-sample-type placeholder folder structure
        generate_folder_structure(samples_dir, sample_types)

        #     TODO 3.1.5 overwrite the matching empty placeholders with existing
        #                masks (skipped when --masks was not given)
        inject_masks(samples_dir, case, mask_dir, mask_types)

        # A case's conversion jobs are grouped: contrast in one group (possibly
        # several series folders, e.g. ea's T1 pre and post) and T2 in another.
        # One job per series folder, converting into its own raw folder
        # _raw_{group}_{idx}. dcmfolder2nrrd decides for itself whether that series
        # folder is already split by subfolder or has to be split by acquisition
        # number, producing c0.nrrd, c1.nrrd ...
        t2_dir = source.get("t2_dir")
        jobs = [
            ("contrast", source.get("contrast_dirs") or [], contrast_types),
            ("t2", [t2_dir] if t2_dir else [], t2_types),
        ]
        for group, dicom_dirs, types in jobs:
            if not dicom_dirs or not types:
                continue
            for idx, dicom_dir in enumerate(dicom_dirs):
                all_temp_cases.append({
                    "name": f"{case} [{group}-{idx}]",
                    # source is the series root folder
                    "source": dicom_dir,
                    # dest is this series' raw output folder; the contents are
                    # distributed to the sample_types afterwards
                    "dest": samples_dir / f"_raw_{group}_{idx}",
                })

        case_dirs.append({
            "name": case,
            "samples_dir": samples_dir,
            "processed_dir": processed_dir,
        })

    #     TODO 3.2 convert the DICOM series to nrrd (parallel, slow)
    processor_for_convert_nrrd(all_temp_cases, use_compression)

    # ---- pass 2: distribute + register copy + reorganize + move into dataset ----
    for case in case_dirs:
        samples_dir = case["samples_dir"]
        processed_dir = case["processed_dir"]

        #     TODO 3.3 distribute the raw cN.nrrd files into their sample_type folders
        distribute_converted_nrrd(samples_dir, "contrast", contrast_types)
        distribute_converted_nrrd(samples_dir, "t2", t2_types)
        #     TODO 3.4 copy every contrast once into its registration placeholder
        copy_contrast_to_register(samples_dir, contrast_types, registration_types)
        #     TODO 3.5 reorganize sample_type folders into sam-N
        format_folder_structure(samples_dir, processed_dir, sample_types)
        #     TODO 3.6 move samples into the dataset and set the sample type
        move_files(processed_dir, sample_types)

    dataset.save()


def processor_for_convert_nrrd(all_temp_cases, use_compression=NRRD_USE_COMPRESSION):
    total = len(all_temp_cases)
    print(f"[convert] {total} conversion job(s), nrrd "
          f"{'compressed' if use_compression else 'uncompressed'}, starting...", flush=True)
    with ProcessPoolExecutor(max_workers=60) as executor:
        future_to_name = {
            executor.submit(dcmfolder2nrrd, temp['source'], temp['dest'], 'c',
                            use_compression): temp['name']
            for temp in all_temp_cases
        }
        done_count = 0
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            done_count += 1
            try:
                future.result()
                print(f"[{done_count}/{total}] done: {name}", flush=True)
            except Exception as e:
                print(f"[{done_count}/{total}] failed: {name} -> {e}", flush=True)
    print(f"[convert] all conversions finished ({total} total)", flush=True)


# TODO 3.1 generate the per-sample-type placeholder folder structure
def generate_folder_structure(samples_dir, sample_types):
    """Create one folder per sample_type with an (empty) placeholder file.

    The contrast / t2 placeholders are overwritten by the DICOM conversion; the
    registration ones by the contrast copy; the *_LPS_nii placeholders stay empty
    until real data exists.
    """
    for sample_type in sample_types:
        sample_dir = samples_dir / sample_type
        sample_dir.mkdir(parents=True, exist_ok=True)
        placeholder = sample_dir / sample_type_to_filename(sample_type)
        if not placeholder.exists():
            placeholder.touch()


# TODO 3.1.5 overwrite the matching sample_type placeholders with existing masks
def inject_masks(samples_dir, case, mask_dir, mask_types=None):
    """Copy the mask belonging to this case from mask_dir into the given
    sample_type folders.

    A file is matched when its *filename contains the case id*, so
    plan/reference/volunteer/VL00001_mask.nii.gz matches case 00001, and ADHB-style
    CL00001_*.nii.gz works the same way -- no prefix is hard-coded. Matching is
    case-insensitive.

    No match prints a warning and keeps the empty placeholder (other cases are
    unaffected); several matches raise, because stopping for a human to confirm is
    better than quietly picking the wrong mask and writing it into the dataset.

    :param samples_dir: this case's samples folder
    :type samples_dir: Path
    :param case: case id, e.g. "00001"
    :type case: str
    :param mask_dir: folder holding the masks; None = no injection, returns
                     immediately
    :type mask_dir: Path | None
    :param mask_types: sample_types to fill, default DEFAULT_MASK_TYPES
    :type mask_types: list[str] | None
    """
    if mask_dir is None:
        return
    mask_dir = Path(mask_dir)
    if not mask_dir.is_dir():
        print(f"Mask folder not found, mask injection skipped: {mask_dir}")
        return
    types = mask_types if mask_types else DEFAULT_MASK_TYPES

    matches = sorted(p for p in mask_dir.iterdir()
                     if p.is_file() and case.lower() in p.name.lower())
    if not matches:
        print(f"No mask for case {case} in {mask_dir}, keeping the empty placeholder")
        return
    if len(matches) > 1:
        raise ValueError(
            f"case {case} matches {len(matches)} masks in {mask_dir}: "
            f"{[p.name for p in matches]}. Confirm which one to use (or tidy up "
            f"the mask folder).")

    mask_file = matches[0]
    for sample_type in types:
        expected = sample_type_to_filename(sample_type)
        if not mask_file.name.endswith(expected.split(".", 1)[1]):
            print(f"Warning: the extension of {mask_file.name} does not match the "
                  f"{expected} expected by {sample_type}; writing it as {expected} anyway")
        target_dir = samples_dir / sample_type
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / sample_type_to_filename(sample_type)
        shutil.copy2(mask_file, target)
    print(f"[mask] case {case}: {mask_file.name} -> {types}")


# TODO 3.3 distribute the converted cN.nrrd files into their sample_type folders
def distribute_converted_nrrd(samples_dir, group, types):
    """dcmfolder2nrrd turns each series folder into
    _raw_{group}_{n}/c0.nrrd, c1.nrrd ... This concatenates the contrasts from all
    raw folders by (folder index, contrast index), then moves them in order into
    the sample_type folders of types[0], types[1] ... renaming them to the proper
    file name.

    volunteer / adhb have a single raw folder (c0..cN are all the contrasts); ea
    has two raw folders producing one c0 each, which concatenate into pre + post.
    When more contrasts are converted than there are types, the extras have no
    matching sample_type and are dropped."""
    if not types:
        return
    raw_dirs = sorted((p for p in samples_dir.glob(f"_raw_{group}_*") if p.is_dir()),
                      key=lambda p: int(p.name.rsplit("_", 1)[1]))
    if not raw_dirs:
        print(f"No converted {group} nrrd in {samples_dir}")
        return

    converted = []
    for raw_dir in raw_dirs:
        converted.extend(sorted(raw_dir.glob("c*.nrrd"),
                                key=lambda p: int(p.stem[1:])))
    if len(converted) > len(types):
        print(f"{samples_dir}: {group} produced {len(converted)} contrast(s), "
              f"keeping only the first {len(types)}")

    for idx, sample_type in enumerate(types):
        if idx >= len(converted):
            break
        source_file = converted[idx]
        if is_empty_file(source_file):
            continue
        target_dir = samples_dir / sample_type
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / sample_type_to_filename(sample_type)
        if target.exists():
            target.unlink()
        source_file.rename(target)

    # Whatever is left in the raw folders is empty or surplus; clear it out
    for raw_dir in raw_dirs:
        shutil.rmtree(raw_dir, ignore_errors=True)


# TODO 3.4 copy every converted contrast nrrd to its registration placeholder
def copy_contrast_to_register(samples_dir, contrast_types, registration_types):
    """Copy each contrast once into its registration placeholder, to be overwritten
    by the actual registration later."""
    for contrast_type, register_type in zip(contrast_types, registration_types):
        contrast_file = samples_dir / contrast_type / sample_type_to_filename(contrast_type)
        if is_empty_file(contrast_file):
            print(f"No converted nrrd for {contrast_type}, register copy skipped")
            continue
        register_dir = samples_dir / register_type
        register_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contrast_file, register_dir / sample_type_to_filename(register_type))


# TODO 3.5 re-organize sample_type folders into sam-N (order follows sample_types)
def format_folder_structure(samples_dir, processed_dir, sample_types):
    for idx, sample_type in enumerate(sample_types, start=1):
        filename = sample_type_to_filename(sample_type)
        src_file = samples_dir / sample_type / filename
        if not src_file.exists():
            print(f"Sample file missing, skip: {src_file}")
            continue
        # Drop empty nrrd (failed conversion); keep empty .nii placeholders.
        if src_file.suffix == ".nrrd" and is_empty_file(src_file):
            print(f"Empty nrrd, skip: {src_file}")
            continue
        dst = processed_dir / f"sam-{idx}" / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst)


# TODO 3.6 Move files to sds dataset and set each sample's type
def move_files(processed_dir, sample_types):
    global dataset
    assert isinstance(dataset, Dataset)
    if dataset is None:
        return
    subdirs = [p for p in processed_dir.iterdir() if p.is_dir()]
    subdirs_sorted = sorted(subdirs, key=lambda x: int(x.name.split("-")[1]))
    subject = Subject()
    for subdir in subdirs_sorted:
        # sam-N maps to sample_types[N-1] so the type stays correct even if some
        # sam folders were skipped (e.g. a failed conversion).
        sam_index = int(subdir.name.split("-")[1])
        sample_type = sample_types[sam_index - 1]
        sample = Sample()
        sample.add_path(subdir)
        subject.add_samples([sample])
        sample.set_value(element='sample type', value=sample_type)
    dataset.add_subjects([subject])


def _update_dataset_description(title):
    global dataset
    assert isinstance(dataset, Dataset)
    dataset_description = dataset.get_metadata(metadata_file="dataset_description")
    dataset_description.add_values(element='type', values="experimental")
    dataset_description.add_values(element='Title', values=title)
    dataset_description.add_values(element='Keywords', values=["breast cancer", "image processing"])
    dataset_description.set_values(
        element='Contributor orcid',
        values=["https://orcid.org/0000-0001-8170-199X"])
    dataset_description.save()


# ---------------------------------------------------------------------------
# Configuration
#
# Everything that identifies real data -- share paths, case / subject ids --
# comes from .env (see .env.example). Nothing here is a real location.
# ---------------------------------------------------------------------------
# Manually annotated samples, identical across all three workflows
MANUAL_TYPES = ["model_predicted_LPS_nii", "researcher_manual_LPS_nii", "ground_truth_LPS_nii", "findings_json", "vessel_mask_nii"]

# When --masks points at a mask folder, the masks fill these two sample_types by
# default. To fill ground_truth_LPS_nii as well, list it explicitly via --mask-types.
DEFAULT_MASK_TYPES = ["model_predicted_LPS_nii", "researcher_manual_LPS_nii"]

# ---- volunteer: one series folder per case, found by a path template ----
VOLUNTEER_DICOM_ROOT = _env_path("DICOM_ROOT_VOLUNTEER")
# {root} and {case} are substituted per case. Override in .env for a different layout.
VOLUNTEER_SERIES_TEMPLATE = _env("SERIES_TEMPLATE_VOLUNTEER", "{root}/{case}/MRI_T2_prone")
VOLUNTEER_CASES = _env_list("CASES_VOLUNTEER")
VOLUNTEER_CONTRAST_TYPES = ["contrast_pre_nrrd"]
VOLUNTEER_REGISTRATION_TYPES = ["registration_pre_nrrd"]
VOLUNTEER_T2_TYPES = []

# ---- adhb: DICOM paths come from a manifest json ----
ADHB_METADATA_PATH = _env_path("MANIFEST_PATH_ADHB", SCRIPT_DIR / "data" / "manifest_adhb.json")
# Empty = every case in the json that has a path (entries that are empty objects
# are skipped automatically)
ADHB_CASES = _env_list("CASES_ADHB")
ADHB_CONTRAST_TYPES = ["contrast_pre_nrrd", "contrast_1_nrrd", "contrast_2_nrrd",
                       "contrast_3_nrrd", "contrast_4_nrrd"]
ADHB_REGISTRATION_TYPES = ["registration_pre_nrrd", "registration_1_nrrd", "registration_2_nrrd",
                           "registration_3_nrrd", "registration_4_nrrd"]
ADHB_T2_TYPES = ["t2_nrrd"]

# The T2 path sits in latest.t2 of the same json (currently the same as t1,
# pending an update). Set this to False to skip T2; the t2_* samples stay empty.
ADHB_WITH_T2 = True

# ---- ea: the EA1141 public dataset; DICOM paths inferred from the Modality /
#      StudyDate headers ----
EA_ROOT = _env_path("DICOM_ROOT_EA")
# Empty = every subject under EA_ROOT
EA_SUBJECTS = _env_list("CASES_EA")
# One independent dataset per timepoint: 0 = first MRI exam, 1 = second
EA_TIMEPOINTS = [0, 1]
# EA only converts the two T1 phases: pre and post
EA_CONTRAST_TYPES = ["contrast_pre_nrrd", "contrast_1_nrrd"]
EA_REGISTRATION_TYPES = ["registration_pre_nrrd", "registration_1_nrrd"]
EA_T2_TYPES = ["t2_nrrd"]

# Same meaning as ADHB_WITH_T2.
#   False: do not convert T2, leaving an empty t2_nrrd placeholder in samples/.
#          Empty nrrds are dropped in format_folder_structure, so the final SDS has
#          no sam folder for t2.
#   True:  find the T2 series in the exam by SeriesDescription and convert it into
#          t2_nrrd
EA_WITH_T2 = True

# Every generated dataset lands here (already ignored by .gitignore), under its own
# timestamped name, so runs never overwrite each other. Intermediates go into a
# _work/ folder of the same name and can be deleted wholesale afterwards.
OUT_ROOT = SCRIPT_DIR / "generated_datasets"

WORKFLOW_CONFIGS = {
    "volunteer": {
        "title": "Volunteer MRI dataset",
        "contrast_types": VOLUNTEER_CONTRAST_TYPES,
        "registration_types": VOLUNTEER_REGISTRATION_TYPES,
        "t2_types": VOLUNTEER_T2_TYPES,
        "sample_types": VOLUNTEER_CONTRAST_TYPES + VOLUNTEER_REGISTRATION_TYPES
                        + VOLUNTEER_T2_TYPES + MANUAL_TYPES,
    },
    "adhb": {
        "title": "ADHB MRI dataset",
        "contrast_types": ADHB_CONTRAST_TYPES,
        "registration_types": ADHB_REGISTRATION_TYPES,
        "t2_types": ADHB_T2_TYPES,
        "sample_types": ADHB_CONTRAST_TYPES + ADHB_REGISTRATION_TYPES
                        + ADHB_T2_TYPES + MANUAL_TYPES,
    },
    "ea": {
        "title": "EA1141 MRI dataset",
        "contrast_types": EA_CONTRAST_TYPES,
        "registration_types": EA_REGISTRATION_TYPES,
        "t2_types": EA_T2_TYPES,
        "sample_types": EA_CONTRAST_TYPES + EA_REGISTRATION_TYPES
                        + EA_T2_TYPES + MANUAL_TYPES,
    },
}

# Default case list per workflow, used when --cases is not given
DEFAULT_CASES = {
    "volunteer": VOLUNTEER_CASES,
    "adhb": ADHB_CASES,
    "ea": EA_SUBJECTS,
}

# Default {root} substitution per workflow, used by --template when --root is omitted
DEFAULT_ROOTS = {
    "volunteer": VOLUNTEER_DICOM_ROOT,
    "adhb": None,          # adhb paths are absolute in the manifest
    "ea": EA_ROOT,
}

# Default T2 switch per workflow, used when neither --with-t2 nor --no-t2 is given
DEFAULT_WITH_T2 = {
    "volunteer": False,   # volunteer has no T2 to begin with
    "adhb": ADHB_WITH_T2,
    "ea": EA_WITH_T2,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert DICOM into a SPARC SDS dataset "
                    "(three workflows: volunteer / adhb / ea)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # First time with an unfamiliar dataset: see how it is actually laid out and
  # write a manifest (read-only, converts nothing)
  python main.py --discover /path/to/dicom_root

  # Run three volunteer cases and fill existing masks into
  # model_predicted / researcher_manual
  python main.py --workflow volunteer --cases 00001 00002 00003 \\
                 --masks /path/to/masks

  # Any fixed layout, without touching the code
  python main.py --template "{root}/{case}/T1_pre" \\
                 --template "{root}/{case}/T1_post" \\
                 --root /path/to/dicom --cases CASE-01

  # Run ea (produces two datasets in one go, timepoint 1 and 2)
  python main.py --workflow ea --cases SUBJECT-0001

  # No arguments = use the values from .env
  python main.py

Data locations and case ids come from .env; copy .env.example to get started.
""")
    parser.add_argument("--discover", metavar="DIR", default=None,
                        help="read-only: report the DICOM layout under DIR and "
                             "write a manifest, then exit. Converts nothing")
    parser.add_argument("--discover-out", type=Path, default=None,
                        help="where --discover writes its manifest "
                             "(default data/discovered_manifest.json)")
    parser.add_argument("--template", action="append", default=None, metavar="TPL",
                        help="path template with {root} and {case} placeholders, "
                             "e.g. \"{root}/{case}/T1_pre\". Repeat the flag when "
                             "the contrasts live in separate folders; they map "
                             "onto contrast types in order. Overrides --workflow's "
                             "own way of locating folders")
    parser.add_argument("--t2-template", default=None, metavar="TPL",
                        help="path template for the T2 series, used with --template")
    parser.add_argument("--root", default=None,
                        help="value substituted for {root} in the templates "
                             "(default: the workflow's root from .env)")
    parser.add_argument("--workflow", choices=list(WORKFLOW_CONFIGS), default=WORKFLOW,
                        help=f"which workflow to run (default {WORKFLOW})")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="run only these cases/subjects (default: the "
                             "workflow's constant list)")
    parser.add_argument("--masks", type=Path, default=None,
                        help="folder holding existing masks; matched by the case "
                             "id appearing in the filename. Omit to skip "
                             "injection and leave the placeholders empty")
    parser.add_argument("--mask-types", nargs="+", default=None,
                        help=f"which sample_types the masks fill "
                             f"(default {' '.join(DEFAULT_MASK_TYPES)})")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT,
                        help=f"root directory for every dataset "
                             f"(default {OUT_ROOT.name}/)")
    parser.add_argument("--name", default=None,
                        help="dataset name for this run (default "
                             "{workflow}_{YYYYmmdd-HHMMSS}). ea appends "
                             "_tp1 / _tp2 to it")
    parser.add_argument("--force", action="store_true",
                        help="delete and rebuild when the target name already "
                             "exists; the default is to abort without overwriting")
    t2 = parser.add_mutually_exclusive_group()
    t2.add_argument("--with-t2", dest="with_t2", action="store_true", default=None,
                    help="force T2 conversion on")
    t2.add_argument("--no-t2", dest="with_t2", action="store_false",
                    help="force T2 conversion off")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()

    # --discover is a read-only inspection mode: report and exit, convert nothing.
    if args.discover:
        discover_dicom_layout(
            args.discover,
            args.discover_out or SCRIPT_DIR / "data" / "discovered_manifest.json")
        raise SystemExit(0)

    workflow = args.workflow
    config = WORKFLOW_CONFIGS[workflow]
    cases = args.cases if args.cases else DEFAULT_CASES[workflow]
    with_t2 = args.with_t2 if args.with_t2 is not None else DEFAULT_WITH_T2[workflow]

    # ea produces one independent dataset per timepoint (<name>_tp1 / <name>_tp2);
    # volunteer and adhb run once.
    if workflow == "ea":
        runs = [{"timepoint": tp, "suffix": f"_tp{tp + 1}"} for tp in EA_TIMEPOINTS]
    else:
        runs = [{"timepoint": None, "suffix": ""}]

    # Base name for this run. Timestamped by default, so every run is a fresh copy
    # and never overwrites an earlier result.
    base_name = args.name if args.name else \
        f"{workflow}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_root = Path(args.out_root)

    # Resolve every target directory and check for name clashes before converting
    # anything: better to run nothing at all than to finish timepoint 1 and only
    # then discover that timepoint 2 clashes.
    for run in runs:
        target = out_root / f"{base_name}{run['suffix']}"
        if target.exists() and any(target.iterdir()):
            if not args.force:
                raise SystemExit(
                    f"Target dataset already exists and is not empty: {target}\n"
                    f"Pick a different --name, or rerun with --force once you are "
                    f"sure you want to discard it.")
            print(f"--force: overwriting the existing {target}", flush=True)
        run["save_dir"] = target
        run["work_dir"] = out_root / "_work" / f"{base_name}{run['suffix']}"

    for run in runs:
        save_dir = run["save_dir"]
        self_dest_dir = run["work_dir"]
        title = config["title"]
        if run["timepoint"] is not None:
            title = f"{title} (timepoint {run['timepoint'] + 1})"
        self_dest_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)

        # The only difference between the workflows: how a case's DICOM folders
        # are located. --template overrides all of them with an explicit layout.
        if args.template:
            root = args.root if args.root is not None else DEFAULT_ROOTS.get(workflow)
            case_sources = build_template_sources(
                args.template, require(cases, "--cases", "the case ids to run"),
                root, args.t2_template)
        elif workflow == "volunteer":
            case_sources = build_template_sources(
                [VOLUNTEER_SERIES_TEMPLATE],
                require(cases, "CASES_VOLUNTEER", "the case ids to run"),
                require(VOLUNTEER_DICOM_ROOT, "DICOM_ROOT_VOLUNTEER",
                        "the folder holding the per-case DICOM directories"))
        elif workflow == "adhb":
            case_sources = build_manifest_sources(ADHB_METADATA_PATH, cases, with_t2)
        else:
            case_sources = build_ea_sources(
                require(EA_ROOT, "DICOM_ROOT_EA",
                        "the EA1141 root holding the subject folders"),
                require(cases, "CASES_EA", "the subject ids to run"),
                run["timepoint"], with_t2)
        print(f"[{save_dir.name}] {len(case_sources)} case(s) to process -> {save_dir}",
              flush=True)

        # create dataset
        createWorkflowDataset(save_dir, title)

        # build the SDS dataset end-to-end (convert -> distribute -> register -> format -> move)
        generate_self_data_structure(case_sources, self_dest_dir, config,
                                     NRRD_USE_COMPRESSION, args.masks, args.mask_types)
