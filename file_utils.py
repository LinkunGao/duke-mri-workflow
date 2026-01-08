from pathlib import Path
import shutil


def delete_folder(folder_path):
    folder_path = Path(folder_path)
    # Check if the folder exists
    if folder_path.exists() and folder_path.is_dir():
        # Use shutil.rmtree to delete the folder and its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents deleted successfully.")
    else:
        print(f"Folder '{folder_path}' does not exist.")


def delete_zero_size_nrrd_files(folder_path):
    if folder_path.exists() and folder_path.is_dir():
        nrrd_files = folder_path.rglob("*")
        for file in nrrd_files:
            if file.is_file() and file.stat().st_size == 0:
                print("delete：" + file.name)
                file.unlink()


def get_first_file(parent_folder):
    # Check if the provided path is a directory
    first_child = next(parent_folder.iterdir(), None)

    if first_child is not None:
        if first_child.is_file():
            # Return the path of the first file
            return first_child
        else:
            print(f"No files found in '{parent_folder}'.")
    else:
        print(f"The path '{parent_folder}' is not a directory.")


def get_first_child(path, cases_paths):
    first_child = next(path.iterdir(), None)
    if first_child is None:
        print(f"Not found child file in {path}")
    elif first_child.is_dir():
        get_first_child(first_child, cases_paths)
    elif first_child.is_file():
        cases_paths.append(path)


def find_file(directory_path, filename):
    # Convert the input to a Path object
    directory_path = Path(directory_path)
    # Iterate over the contents of the directory
    for item in directory_path.iterdir():
        # Check if it's a directory
        if item.is_dir():
            # Recursively call the function for subdirectories
            result = find_file(item, filename)
            if result:
                return result  # Return the result if found in a subdirectory
        elif item.is_file() and item.name == filename:
            return item  # Return the Path object if the file is found

    # Return None if no 'mask.json' file is found in the directory or its subdirectories
    return None


def is_empty_file(p: Path):
    return (not p.exists()) or (p.stat().st_size == 0)


def rebuild_sam_dirs(case_root_path: Path, prefix: str, filename_pattern: str):
    """
    prefix: 'sam'
    filename_pattern: 'c{}.nrrd' or 'r{}.nrrd'
    """

    # 1. get all sam-* dir
    sam_dirs = sorted(case_root_path.glob(f"{prefix}-*"), key=lambda x: int(x.name.split('-')[1]))

    # 2. get all not empty file
    valid_items = []
    for sam_dir in sam_dirs:
        index = int(sam_dir.name.split('-')[1])
        file_path = sam_dir / filename_pattern.format(index - 1)  # c0,nrrd corresponding to sam-1

        if not is_empty_file(file_path):
            valid_items.append(file_path)

    # 3. delete all old sam folders
    for sam_dir in sam_dirs:
        shutil.rmtree(sam_dir, ignore_errors=True)

    # 4. recreate sam-i by the new order
    for new_idx, old_file in enumerate(valid_items, start=1):
        new_dir = case_root_path / f"{prefix}-{new_idx}"
        new_dir.mkdir(parents=True, exist_ok=True)

        new_filename = filename_pattern.format(new_idx - 1)
        shutil.copy(old_file, new_dir / new_filename)
