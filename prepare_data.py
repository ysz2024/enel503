import os
import shutil

SRC_DATA = "dataset"
MODEL1_DIR = "model1_data"
MODEL2_DIR = "model2_data"
MODEL3_DIR = "model3_data"

def copy_data(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

if __name__ == "__main__":
    copy_data(SRC_DATA, MODEL1_DIR)
    copy_data(SRC_DATA, MODEL2_DIR)
    copy_data(SRC_DATA, MODEL3_DIR)
    print("Copied dataset into 3 folders.")
