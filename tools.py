from pathlib import Path
import shutil
import SimpleITK as sitk
import pydicom
import numpy as np
import os


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

def convertor(dcm_path, nrrd_path):
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_series)
    image = reader.Execute()
    sitk.WriteImage(image, nrrd_path, useCompression=True)
def dcmseries2nrrd(source, dest, filename):
    '''
    A python script for convert dicom files to nrrd file with sepreating contrast.
    If your dicom files include 5 different contrast images, using this script,
    you will get 5 nrrd files with different contrast!
    Python version: v3.9.0
    :Dependency: pip install SimpleITK
    :Author: Linkun Gao
    '''

    reader = sitk.ImageSeriesReader()
    for idx, s in enumerate(source.keys()):
        datapath = source[s].as_posix()
        dicom_series = reader.GetGDCMSeriesFileNames(datapath)
        reader.SetFileNames(dicom_series)
        image = reader.Execute()
        if not dest.exists():
            # Create the folder if it doesn't exist
            dest.mkdir(parents=True, exist_ok=True)
        name = dest / (filename + str(idx) + '.nrrd')
        sitk.WriteImage(image, name, useCompression=True)


def convert_nii_to_nrrd(source, dest, origin_pre_nrrd):
    source = [r'./import/reg_contrast_0-1.nii.gz', r'./import/reg_contrast_0-2.nii.gz',
              r'./import/reg_contrast_0-3.nii.gz', r'./import/reg_contrast_0-4.nii.gz']
    dest = [r'./export/r1.nrrd',r'./export/r2.nrrd',r'./export/r3.nrrd',r'./export/r4.nrrd']

    pre_image = sitk.ReadImage(origin_pre_nrrd)
    for i in range(len(source)):

        input_image = sitk.ReadImage(source[i])
        input_image.CopyInformation(pre_image)
        sitk.WriteImage(input_image, dest[i])