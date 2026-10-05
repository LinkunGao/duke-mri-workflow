from pathlib import Path
import shutil
import SimpleITK as sitk
import pydicom
import numpy as np
import os
import re


def getDicomInfo(dicom_path):
    """
    (0020, 000d) Study Instance UID
    (0020, 000e) Series Instance UID
    :param dicom_path:
    :return:
    """

    dicom_file = pydicom.dcmread(dicom_path)
    # print(dicom_file)

    dicomInfo = {
        "study_uid": f"sub-{dicom_file[(0x0020, 0x000d)].value}",
        "series_uid": f"sam-{dicom_file[(0x0020, 0x000e)].value}"
    }

    return dicomInfo

# Default for nrrd compression. False = write uncompressed nrrd (the default),
# True = gzip compressed. Callers (main.py) can override per call via use_compression.
DEFAULT_USE_COMPRESSION = False


def _fmt_intensity(value):
    """Format min/max for the nrrd header. Integer volumes render as 3095, not 3095.0."""
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def write_nrrd(image, path, use_compression=DEFAULT_USE_COMPRESSION):
    """Write a nrrd and record this volume's intensity min / max in the header.

    ITK renders metadata as NRRD key/value fields, so the header looks like::

        encoding: raw
        max:=3095
        min:=-111

    Reading it back gives image.GetMetaData("min") / ("max"), so downstream
    consumers (registration, front-end window/level) do not have to rescan the
    whole volume just to learn its intensity range.

    :param image: the image to write out
    :type image: sitk.Image
    :param path: output nrrd path
    :type path: Path | str
    :param use_compression: whether to compress, default False (uncompressed)
    :type use_compression: bool
    :return: (min, max), handy for the caller's logging
    :rtype: tuple[float, float]
    """
    stats = sitk.MinimumMaximumImageFilter()
    stats.Execute(image)
    intensity_min, intensity_max = stats.GetMinimum(), stats.GetMaximum()
    image.SetMetaData("min", _fmt_intensity(intensity_min))
    image.SetMetaData("max", _fmt_intensity(intensity_max))
    sitk.WriteImage(image, str(path), useCompression=use_compression)
    return intensity_min, intensity_max


def convertor(dcm_path, nrrd_path, use_compression=DEFAULT_USE_COMPRESSION):
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_series)
    image = reader.Execute()
    write_nrrd(image, nrrd_path, use_compression)
def dcmseries2nrrd(source, dest, filename, use_compression=DEFAULT_USE_COMPRESSION):
    '''
    A python script for convert dicom files to nrrd file with sepreating contrast.
    If your dicom files include 5 different contrast images, using this script,
    you will get 5 nrrd files with different contrast!
    Python version: v3.9.0
    :Dependency: pip install SimpleITK
    :Author: Linkun Gao
    '''

    reader = sitk.ImageSeriesReader()
    total_contrasts = len(source)
    case_label = dest.parts[-4] if len(dest.parts) >= 4 else str(dest)
    print(f"  [{case_label}] starting, {total_contrasts} contrast(s) total", flush=True)
    for idx, s in enumerate(source.keys()):
        datapath = source[s].as_posix()
        print(f"  [{case_label}] ({idx+1}/{total_contrasts}) reading DICOM: {s}  path: {datapath}", flush=True)
        dicom_series = reader.GetGDCMSeriesFileNames(datapath)
        print(f"  [{case_label}] ({idx+1}/{total_contrasts}) found {len(dicom_series)} DICOM files, parsing...", flush=True)
        reader.SetFileNames(dicom_series)
        image = reader.Execute()
        size = image.GetSize()
        print(f"  [{case_label}] ({idx+1}/{total_contrasts}) image size: {size}, writing NRRD...", flush=True)
        if not dest.exists():
            dest.mkdir(parents=True, exist_ok=True)
        name = dest / (filename + str(idx) + '.nrrd')
        intensity_min, intensity_max = write_nrrd(image, name, use_compression)
        file_size_mb = name.stat().st_size / (1024 * 1024)
        print(f"  [{case_label}] ({idx+1}/{total_contrasts}) written: {name.name}  "
              f"({file_size_mb:.1f} MB, min/max {_fmt_intensity(intensity_min)}/"
              f"{_fmt_intensity(intensity_max)})", flush=True)
    print(f"  [{case_label}] all contrasts converted", flush=True)


def dcmfolder2nrrd(source, dest, filename, use_compression=DEFAULT_USE_COMPRESSION):
    '''
    Given one series root folder, convert every contrast in that series into:
        dest/{filename}0.nrrd, dest/{filename}1.nrrd ...

    Two layouts are detected automatically:
      1. The root folder contains subfolders -- the contrasts have already been
         split up (one subfolder per contrast). Sort by subfolder name and convert
         one nrrd per subfolder; loose dcm files in the root are ignored.
      2. The root folder contains only dcm files -- read them and group by
         Acquisition Number (0020|0012). Several contrasts mixed together are split
         into several nrrds (ADHB dynamic series: 560 files = 5 contrasts x 112
         slices); a single group (volunteer's single-phase T2) yields one nrrd.

    :param source: series root folder
    :type source: Path
    :param dest: output folder for the nrrds
    :type dest: Path
    :param filename: output filename prefix, producing {filename}{idx}.nrrd
    :type filename: str
    :param use_compression: whether to compress the nrrd, default False
                            (uncompressed: larger files, faster read/write)
    :type use_compression: bool
    '''
    source = Path(source)
    case_label = dest.parts[-3] if len(dest.parts) >= 3 else str(dest)
    reader = sitk.ImageSeriesReader()

    sub_dirs = sorted((p for p in source.iterdir() if p.is_dir()), key=_dir_sort_key)
    if sub_dirs:
        # Case 1: someone has already split the contrasts into separate subfolders
        print(f"  [{case_label}] {source} has {len(sub_dirs)} subfolder(s), "
              f"splitting contrasts by subfolder: {[p.name for p in sub_dirs]}", flush=True)
        groups = [reader.GetGDCMSeriesFileNames(p.as_posix()) for p in sub_dirs]
        reverse = False
    else:
        # Case 2: dcm files sit directly in the root; use acquisition number to
        # find out whether several contrasts are mixed together
        dcms_name = reader.GetGDCMSeriesFileNames(source.as_posix())
        print(f"  [{case_label}] {source} has {len(dcms_name)} DICOM file(s), "
              f"grouping by acquisition number...", flush=True)
        groups, reverse = _group_by_acquisition_number(dcms_name)
        print(f"  [{case_label}] split into {len(groups)} contrast(s), "
              f"{[len(g) for g in groups]} file(s) each", flush=True)

    dest.mkdir(parents=True, exist_ok=True)
    for idx, dcm_files in enumerate(groups):
        if not dcm_files:
            print(f"  [{case_label}] ({idx+1}/{len(groups)}) no DICOM, skipped", flush=True)
            continue
        if reverse:
            dcm_files = list(dcm_files)[::-1]
        reader.SetFileNames(dcm_files)
        image = reader.Execute()
        name = dest / (filename + str(idx) + '.nrrd')
        intensity_min, intensity_max = write_nrrd(image, name, use_compression)
        file_size_mb = name.stat().st_size / (1024 * 1024)
        print(f"  [{case_label}] ({idx+1}/{len(groups)}) {name.name} size {image.GetSize()} "
              f"({file_size_mb:.1f} MB, {'compressed' if use_compression else 'uncompressed'}, "
              f"min/max {_fmt_intensity(intensity_min)}/{_fmt_intensity(intensity_max)})", flush=True)
    print(f"  [{case_label}] all contrasts converted", flush=True)


def _group_by_acquisition_number(dcms_name):
    """Group a pile of dcm files by Acquisition Number (0020|0012); one group is
    one contrast. A single acquisition number yields a single group (volunteer's
    single-phase series).

    :return: (groups sorted by acquisition number, whether each group's slice
              order has to be reversed)
    """
    groups = {}
    acq_times = {}
    # Header only, no pixel data: far faster than sitk.ReadImage over a network share
    for dcm in dcms_name:
        header = pydicom.dcmread(dcm, stop_before_pixels=True,
                                 specific_tags=['AcquisitionNumber', 'AcquisitionTime'])
        acq_number = int(getattr(header, 'AcquisitionNumber', None) or 1)
        groups.setdefault(acq_number, []).append(dcm)
        acq_times.setdefault(acq_number, []).append(float(getattr(header, 'AcquisitionTime', None) or 0))

    keys = sorted(groups)
    # Decreasing acquisition times in the first group mean the slice order is
    # flipped, so every group has to be reversed
    first_times = acq_times[keys[0]]
    reverse = (first_times[-1] - first_times[0]) < 0
    return [groups[key] for key in keys], reverse


def first_dicom_header(folder, tags, max_tries=5):
    """Find the first readable dcm in a folder (recursively) and read only the
    requested header tags.

    Folder naming in public datasets such as EA1141 is completely inconsistent
    (an MRI exam folder may be called "MR BREAST RESEARCH EXAM", "MRIBB3" or
    "ECOG-ACRIN"; a mammogram one "MMSCRCOMBO" or "Standard Screening - Combo"),
    so names are not a reliable signal. Use the Modality / StudyDate header
    values instead.

    :param folder: folder to probe (searched recursively for dcm files)
    :type folder: Path | str
    :param tags: read only these tags, e.g. ["Modality", "StudyDate"]
    :type tags: list[str]
    :param max_tries: how many unreadable dcm files to try before giving up
    :type max_tries: int
    :return: {tag: value}, or None if not a single file could be read
    :rtype: dict | None
    """
    folder = Path(folder)
    tried = 0
    for dcm in folder.rglob("*.dcm"):
        try:
            header = pydicom.dcmread(dcm, stop_before_pixels=True, specific_tags=list(tags))
        except Exception as e:
            tried += 1
            print(f"  Cannot read DICOM header {dcm}: {e}", flush=True)
            if tried >= max_tries:
                break
            continue
        return {tag: getattr(header, tag, None) for tag in tags}
    return None


def _dir_sort_key(path):
    """Sort subfolders by the number in their name: 0, 1, 2, ... 10 (names
    without a number sort last)."""
    match = re.search(r"(\d+)$", path.name)
    return (0 if match else 1, int(match.group(1)) if match else 0, path.name)


def convert_nii_to_nrrd(source, dest, origin_pre_nrrd):
    source = [r'./import/reg_contrast_0-1.nii.gz', r'./import/reg_contrast_0-2.nii.gz',
              r'./import/reg_contrast_0-3.nii.gz', r'./import/reg_contrast_0-4.nii.gz']
    dest = [r'./export/r1.nrrd',r'./export/r2.nrrd',r'./export/r3.nrrd',r'./export/r4.nrrd']

    pre_image = sitk.ReadImage(origin_pre_nrrd)
    for i in range(len(source)):

        input_image = sitk.ReadImage(source[i])
        input_image.CopyInformation(pre_image)
        write_nrrd(input_image, dest[i])
