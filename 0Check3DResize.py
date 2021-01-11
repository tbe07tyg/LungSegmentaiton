import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import ndimage

plt.rcParams.update({'font.size': 22})
def extract_series(series_path):
    img_obj = nib.load(series_path)
    # print("series:", img_obj.shape)
    print("img.get_data_dtype() == np.dtype(np.int16):", img_obj.get_data_dtype() == np.dtype(np.int16))
    # print("img.affine.shape:", img_obj.affine.shape)
    data = img_obj.get_fdata()
    # print("data in numpy:", data.shape)
    print("data in numpy range: [{}, {}]".format(data.min(), data.max()))
    return data

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 256
    desired_width = 256
    desired_height = 256
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


if __name__ == '__main__':
    train_input_series_paths = []
    train_labels_series_paths = []
    val_input_series_paths = []
    val_labels_series_paths = []
    train_total_series = glob('E:\\dataset\\Lung\\COVID-19-20\\COVID-19-20_v2\\Train/*')
    val_total_series = glob('E:\\dataset\\Lung\\COVID-19-20\\COVID-19-20_v2\\Validation/*')
    print("total len of train series:", len(train_total_series))
    print("total len of val series:", len(val_total_series))
    for each_train_path in train_total_series:
        # print(each_train_path)
        if "seg" in each_train_path:
            train_labels_series_paths.append(each_train_path)
        else:
            train_input_series_paths.append(each_train_path)

    for each_val_paths in val_total_series:
        if "seg" in each_val_paths:
            val_labels_series_paths.append(each_val_paths)
        else:
            val_input_series_paths.append(each_val_paths)

    print("total len of train input series:", len(train_input_series_paths))
    print("total len of train label series:", len(train_labels_series_paths))
    print("total len of val input series:", len(val_input_series_paths))
    print("total len of val label series:", len(val_labels_series_paths))

    #
    for i, each in enumerate(train_input_series_paths):
        print(i)
        print(each)
        each_data =  extract_series(each)
        print("raw extractd data shape:", each_data.shape)

        # 3D resize:
        resized_volume = resize_volume(each_data)
        print("resized data shape:", resized_volume.shape)