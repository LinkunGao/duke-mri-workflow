# duke-mri-workflow

## Duke MRI Workflow for SPARC Dataset Generation

This project provides an automated pipeline for processing Duke breast MRI DICOM images and packaging them into a SPARC-compliant dataset.

**Core Workflow:**

1. **DICOM to NRRD Conversion** — Raw DICOM images are converted to NRRD format, organized by contrast phase (e.g., pre-contrast, contrast 1, contrast 2, etc.).

2. **SPARC Dataset Construction** — The converted files are structured into a SPARC dataset using the `sparc_me` library. The pipeline leverages `Subject()` and `Sample()` objects to register each file's path, triggering automated file transfer and updating the dataset manifests (`manifest.xlsx`, `samples.xlsx`, `subjects.xlsx`). See `move_files()` for implementation details.

3. **Sample Type Convention** — Each file must be assigned a `sample_type` that reflects its origin and processing stage. The following conventions apply:

   | File Origin | `sample_type` |
   |---|---|
   | Original DICOM, pre-contrast | `contrast_pre` |
   | Original DICOM, contrast phase 1 | `contrast_1` |
   | Registered image, pre-contrast | `registration_pre` |
   | Registered image, contrast phase N | `registration_N` |
   | Model-predicted segmentation (NIfTI) | `model_predicted_nii` |
   | Researcher manual segmentation mask (NIfTI) | `researcher_manual_nii` |

> **Important:** Always verify and update the `sample_type` field when adding new files to the dataset. Incorrect or missing sample types will result in inconsistent manifest entries.
