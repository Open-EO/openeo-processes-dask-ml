from pathlib import Path


def get_file_chunk(tmp_dir_input: Path, start_chunk: int, num_chunks: int):
    all_files = list(tmp_dir_input.glob("*.npy"))
    file_chunk = all_files[start_chunk::num_chunks]
    return file_chunk
