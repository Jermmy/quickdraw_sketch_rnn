from sketch_rnn.config import quick_draw_dir, sketch_label_file, preprocessed_data_dir
from sketch_rnn.utils import load_data_files

data_files = load_data_files()

f = open(data_files[0], 'r')
print(f.readlines()[0:10])
f.close()


print("===========================================")

f = open(data_files[0], 'r')
print(f.readlines()[0:10])
f.close()