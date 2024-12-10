import tarfile
from pathlib import Path

class TarFileReader:
    def __init__(self, tar_file_path):
        self.tar_file_path = tar_file_path

    def get_image(self, image_path):
        try:
            with tarfile.open(self.tar_file_path, 'r') as tar:
                image_file = tar.extractfile(image_path)
                if image_file is None:
                    raise FileNotFoundError(f"Image '{image_path}' not found in tar file.")
                return image_file.read()
        except KeyError:
            print(f"Warning: Image '{image_path}' not found in tar file.")
            return None
