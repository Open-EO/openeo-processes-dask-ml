import os


def make_tmp_folder(dir_path: str = "./test_tmp") -> str:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


def prepare_tmp_folder(
    dir_path: str = "./test_tmp", file_name: str = "file.bin"
) -> tuple[str, str]:
    file_path = dir_path + "/" + file_name
    dir_path = make_tmp_folder(dir_path)
    if os.path.exists(file_path):
        os.remove(file_path)
    return dir_path, file_path


def clear_tmp_folder(dir_path: str = "./test_tmp", file_name: str = "file.bin"):
    for path in os.listdir(dir_path):
        os.remove(dir_path + "/" + path)
    os.rmdir(dir_path)
