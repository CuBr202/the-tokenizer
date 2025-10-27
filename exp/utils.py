import subprocess
import os

def get_total_lines(file_path: str) -> int:
    """
    Given a file name, return the number of lines.
    """
    if not os.path.exists(file_path):
        raise ValueError(f'File {file_path} not found!')
    lines = subprocess.check_output(['wc', '-l', os.path.abspath(file_path)]).decode('utf-8')
    return int(lines.split(' ', maxsplit=1)[0])
