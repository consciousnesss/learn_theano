from learn_theano.utils.s3_download import S3
import subprocess
import os
import glob


def get_nottingham_midi_folder():
    s3 = S3()
    resulting_folder = os.path.join(s3.cache_folder(), "Nottingham")
    if os.path.isdir(resulting_folder):
        return resulting_folder

    print("Unzipping Nottingham midi dataset...")
    zipped = S3().download('datasets/Nottingham.zip')
    subprocess.check_call("unzip %s -d %s" % (zipped, s3.cache_folder()), shell=True)
    assert (os.path.isdir(resulting_folder))
    return resulting_folder


def get_nottingham_dataset():
    folder = get_nottingham_midi_folder()

    return [glob.glob(os.path.join(folder, set_name, '*.mid'))
            for set_name in ['train', 'valid', 'test']]


if __name__ == "__main__":
    f = get_nottingham_midi_folder()
    print(f)
