import subprocess
import platform


def play_midi(filename):
    if platform.system() != "Darwin":
        raise NotImplementedError("Do not know how to play midi on non-mac OS")
    try:
        subprocess.check_output("timidity %s" % filename, shell=True)
    except Exception as ex:
        raise Exception("Can not call timidity - %s. Try installing with 'brew install timidity'" % str(ex))


if __name__ == "__main__":
    from learn_theano.utils.midi_nottingham_dataset import get_nottingham_midi_folder
    import os
    nottingham = get_nottingham_midi_folder()
    play_midi(os.path.join(nottingham, 'train', 'ashover_simple_chords_1.mid'))
