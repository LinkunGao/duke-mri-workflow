import sparc_me as sm
from file_utils import delete_folder, get_first_file, get_first_child, find_file, delete_zero_size_nrrd_files
from tools import getDicomInfo, dcmseries2nrrd
from pathlib import Path
from shutil import copy2
import shutil
from datetime import datetime, timezone
from pprint import pprint
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import psutil

dataset = None
bounding_box_data = None
duke_metadata = None
case_names = []

contrasts = ["ax dyn pre", "ax dyn 1st pass", "ax dyn 2nd pass", "ax dyn 3rd pass", "ax dyn 4th pass"]
cases = []


# TODO 1: create dataset
def createWorkflowDataset(dest_dir):
    global dataset
    delete_folder(dest_dir)
    dataset = sm.Dataset()
    # NOTE: Step2, way1: load dataset from template
    dataset.set_path(dest_dir)
    dataset.create_empty_dataset(version='2.0.0')
    manifest = dataset.get_metadata(metadata_file="manifest")
    manifest.data['patient_id'] = "nan"

    dataset.save()


# TODO 2: get bounding box data
def get_bounding_box_data(data_path):
    global bounding_box_data
    global case_names
    if not data_path.exists():
        return
    with open(data_path, "r") as f:
        bounding_box_data = json.load(f)
    case_names = list(bounding_box_data.keys())
    case_names.sort()


# TODO 3: read DUKE uni breast MRI metadata
def read_duke_breast_metadata(data_path):
    if not data_path.exists():
        return
    df = pd.read_csv(data_path)

    for case in case_names:
        case_df = df[df['Subject ID'] == case]
        temp = {}
        study_uid = None
        for contrast in contrasts:
            contrast_df = case_df[case_df['Series Description'] == contrast]
            if contrast_df.empty:
                continue
            temp[contrast] = contrast_df['Series UID'].iloc[0]
            study_uid = contrast_df["Study UID"].iloc[0]
        cases.append({
            "name": case,
            "study_uid": study_uid,
            "contrasts": temp
        })


# TODO 4: convert Dicom to NRRD
def generate_self_data_structure(dicom_source_dir, self_dest_dir, ignore_dirs=[]):
    """
    :param dicom_source_dir:
    :type dicom_source_dir: Path
    :param self_dest_dir:
    :type self_dest_dir: Path
    :param ignore_dirs: the folders you want to ignore
    :type ignore_dirs: str[]
    :return:
    """
    # delete_folder(nrrd_dest_dir)
    if dicom_source_dir.is_dir():
        for case in cases:
            for c in case["contrasts"].keys():
                case["contrasts"][c] = dicom_source_dir / case["name"] / case["study_uid"] / case["contrasts"][c]
        # get the first dicom file
        all_temp_cases = []
        for idx, case in enumerate(cases):
            case_root_path = self_dest_dir / case["name"] / f"sub-{case['name'].lower()}" / "sam-1"
            temp = {
                # source is dcm folder path
                "source": case["contrasts"],
                # dest is nrrd file path
                "dest": case_root_path / "nrrd" / "origin"
            }
            all_temp_cases.append(temp)
            #
            #     TODO 3.1 generate folder structure
            #
            generate_folder_structure(case_root_path, case["name"])
            #     TODO 3.2 convert nrrd
            #     need to comment the code, after you already have all nrrds generated, this is very slow, use outside processor 3.2 outside
            #     dcmseries2nrrd(temp['source'], temp['dest'], 'c')
            #
            #     TODO 3.3 convert nii register image to nrrd

            #     TODO 3.4 Move files to sds dataset and modify manifest.xlsx file
            move_files_modify_manifest(case_root_path, self_dest_dir, case["name"])

            #     TODO 3.4 if registration images have not ready, let's mock data, this is based on you've already haven origin nrrds
            mock_register_nrrd(temp["dest"], case["name"])
            # TODO 3.5 delete all origin/registration zero size nrrd files
            delete_zero_size_nrrd_files(
                dataset._dataset_path / "derivative" / f"sub-{case['name'].lower()}" / "sam-1" / "nrrd" / "origin")
            delete_zero_size_nrrd_files(
                dataset._dataset_path / "derivative" / f"sub-{case['name'].lower()}" / "sam-1" / "nrrd" / "registration")

        # TODO: 3.2 outside covert nrrd
        # processor_for_convert_nrrd(all_temp_cases)



    else:
        print(f"The dicom_source_dir: {dicom_source_dir} you provide is not a folder!")
        return


def processor_for_convert_nrrd(all_temp_cases):
    with ProcessPoolExecutor(max_workers=60) as executor:
        futures = [
            executor.submit(dcmseries2nrrd, temp['source'], temp['dest'], 'c')
            for temp in all_temp_cases
        ]
        for future in futures:
            try:
                future.result()  # wait for task finish
            except Exception as e:
                print(f"task fail: {e}")


# TODO 3.1 generate folder structure
def generate_folder_structure(case_root_path, patientID):
    # TODO 3.1.1 Create nrrd/origin, nrrd/registration
    contrast0 = case_root_path / "nrrd/origin/c0.nrrd"
    contrast1 = case_root_path / "nrrd/origin/c1.nrrd"
    contrast2 = case_root_path / "nrrd/origin/c2.nrrd"
    contrast3 = case_root_path / "nrrd/origin/c3.nrrd"
    contrast4 = case_root_path / "nrrd/origin/c4.nrrd"

    r0 = case_root_path / "nrrd/registration/r0.nrrd"
    r1 = case_root_path / "nrrd/registration/r1.nrrd"
    r2 = case_root_path / "nrrd/registration/r2.nrrd"
    r3 = case_root_path / "nrrd/registration/r3.nrrd"
    r4 = case_root_path / "nrrd/registration/r4.nrrd"

    # TODO 3.1.2 Create segmentation
    nipple_points_json = case_root_path / "segmentation/nipple_points.json"
    outer_rib_mesh_surface_points_json = case_root_path / "segmentation/outer_rib_mesh_surface_points.json"
    skin_mesh_surface_points_json = case_root_path / "segmentation/skin_mesh_surface_points.json"
    prone_surface_obj = case_root_path / "segmentation/prone_surface.obj"
    tumour_window_json = case_root_path / "segmentation/tumour_window.json"

    # TODO 3.1.3 Create segmentation_manual
    mask_json = case_root_path / "segmentation_manual/mask.json"
    mask_nii = case_root_path / "segmentation_manual/mask.nii.gz"
    mask_obj = case_root_path / "segmentation_manual/mask.obj"
    sphere_points_json = case_root_path / "segmentation_manual/sphere_points.json"
    tumour_timer_study_json = case_root_path / "segmentation_manual/tumour_position_study.json"

    paths_need_create = [contrast0, contrast1, contrast2, contrast3, contrast4, r0, r1, r2, r3, r4, nipple_points_json,
                         mask_json, mask_nii, mask_obj, sphere_points_json, tumour_timer_study_json,
                         outer_rib_mesh_surface_points_json, skin_mesh_surface_points_json, prone_surface_obj,
                         tumour_window_json]

    # TODO 3.1.4 Generate paths

    for path in paths_need_create:
        if not path.exists():
            # Create the directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            # Create the file
            path.touch()

    # TODO 3.1.5 Move the sphere_points

    # nipple_points_source_path = find_file(fr"Y:\sandbox\clin864\wm_project_cl_t1\results\{patientID}",
    #                                       'nipple_points.json')
    # # Specify the source file path
    # if nipple_points_source_path is not None:
    #
    #     source_file_path = Path(nipple_points_source_path)
    #
    #     # Copy the file to the destination folder
    #     destination_file_path = nipple_points_json
    #
    #     if destination_file_path.exists():
    #         # if already has the file, delete it
    #         destination_file_path.unlink()
    #
    #     copy2(source_file_path, destination_file_path)

    # TODO 3.1.6 Generate tumour position
    tumour_data = bounding_box_data[patientID]["origin"]
    tumour_window = {
        "bounding_box_max_point":  tumour_data["bounding_box_max_point"],
        "bounding_box_min_point": tumour_data["bounding_box_min_point"],
        "center": tumour_data["center"],
        "validate": False
    }

    with open(tumour_window_json, "w") as json_file:
        json.dump(tumour_window, json_file)


# TODO 3.4 Move files to sds dataset and modify manifest.xlsx file

def move_files_modify_manifest(sample_root_path, self_dest_dir, patientID):
    global dataset

    if dataset is not None:
        source_dataset_root = self_dest_dir / patientID
        target_dataset_root = dataset._dataset_path / "derivative"
        manifest = dataset.get_metadata(metadata_file="manifest")
        move(sample_root_path, source_dataset_root, target_dataset_root, patientID, manifest)


def move(source_sam, source_dataset_root, target_dataset_root, patientID, manifest):
    if source_sam.is_dir():
        source_sample_files = source_sam.rglob("*")
        for file in source_sample_files:
            if file.is_file():
                target_path = target_dataset_root / file.relative_to(source_dataset_root)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(file), str(target_path))
                _update_manifest(str(target_path), patientID, manifest)


def _update_manifest(sample_file_path, patientID, manifest):
    """
    Update manifest metadata, after remove samples

    :param sample_path: sample path
    :type sample_path: str
    """
    file_path = f"./{Path(sample_file_path.replace(str(dataset._dataset_path), '')[1:]).as_posix()}"

    if Path(sample_file_path).suffix == 'gz' or Path(sample_file_path).suffix == '.gz':
        suffix = "nii.gz"
    else:
        suffix = Path(sample_file_path).suffix.lstrip('.')
    row = {
        'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        'description': f"tumour position & extent reporting app case {patientID}",
        'file type': suffix,
        'patient_id': patientID
    }
    df_manifest = manifest.data
    # check is exist
    if file_path in df_manifest['filename'].values:
        manifest_index = df_manifest.loc[df_manifest['filename'] == file_path].index[0]
    else:
        manifest_row = [file_path] + [float('nan')] * (len(df_manifest.columns) - 1)
        # Create new row
        manifest_index = len(df_manifest)
        df_manifest.loc[manifest_index] = manifest_row

    for element, value in row.items():
        validate_element = find_col_element(element, manifest)
        df_manifest.loc[manifest_index, validate_element] = value

    manifest.save()


def find_col_element(element, metadata):
    elements = metadata.data.columns.tolist()
    matching_indices = metadata.validate_input(element, elements)
    if len(matching_indices) == 1:
        return elements[matching_indices[0]]
    else:
        msg = f"No '{element}' valid element is found! Please find correct element in {metadata.metadata_file}.xlsx file."
        raise KeyError(msg)


def mock_register_nrrd(origin_folder, case_name):
    files = origin_folder.rglob("*.nrrd")

    target_folder = dataset._dataset_path / "derivative"
    for idx, file in enumerate(files):
        target_path = target_folder / Path(*file.parts[2:5]) / "registration" / f"r{idx}.nrrd"
        shutil.copy(file, target_path)


if __name__ == '__main__':
    save_dir = Path("./workdir")
    dicom_source_dir = Path("Z:\projects\clinical_Duke_cancer\manifest-1607053360376\Duke-Breast-Cancer-MRI")
    duke_mri_metadata = Path("Z:\projects\clinical_Duke_cancer\manifest-1607053360376\metadata.csv")
    self_dest_dir = Path("./test2")
    ignore_dirs = ['CL00012_renamed', 'converted_nrrd']
    bounding_box_data_path = Path("./data/anna_cases_results.json")

    # create dataset
    createWorkflowDataset(save_dir)

    get_bounding_box_data(bounding_box_data_path)

    read_duke_breast_metadata(duke_mri_metadata)
    # convert Dicom to NRRD
    generate_self_data_structure(dicom_source_dir, self_dest_dir, ignore_dirs)

    # rename folder name
    # folders = [f for f in self_dest_dir.iterdir() if f.is_dir()]
