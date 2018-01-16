from sketch_rnn.config import quick_draw_dir, sketch_label_file, preprocessed_data_dir
from sketch_rnn.utils import load_data_files
import numpy as np

out = np.array([[[1,2,3],[2,3,4],[3,4,5]], [[0,1,3],[0,2,1],[0,0,1]]])

print(np.mean(out, axis=1))