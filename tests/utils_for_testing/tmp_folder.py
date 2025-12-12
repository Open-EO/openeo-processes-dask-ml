import os


def prepare_tmp_folder(
    dir_path: str = "./test_tmp", file_name: str = "file.bin"
) -> tuple[str, str]:
    file_path = dir_path + "/" + file_name
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(file_path):
        os.remove(file_path)
    return dir_path, file_path


def clear_tmp_folder(dir_path: str = "./test_tmp", file_name: str = "file.bin"):
    file_path = dir_path + "/" + file_name
    os.remove(file_path)
    os.rmdir(dir_path)
