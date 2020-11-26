import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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

def plot_hist(data, axis, x_name,num_of_bins = 1000):
    axis.hist(data, bins=num_of_bins, color='blue', edgecolor='black', alpha=0.1)
    axis.set_title("Histogram")
    axis.set_xlabel(x_name)
    axis.set_ylabel("Percentage of data")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(data)))




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

    # anaylize the training CTs
    # analyzing the training input ------------------------->
    num_slices_in_training_dataset = []
    label_index_list = []
    label_range_list = []
    all_data_list = None
    fig, ax = plt.subplots() # for number of slices in each series
    fig2, ax2 = plt.subplots() # for intensity for all pixels in CT
    fig3, ax3 =  plt.subplots() # for intensity of annotated pixels in all CTs
    fig4, ax4 = plt.subplots()  # for intensity of standardized data
    for i, each in enumerate(train_input_series_paths):
        print(i)
        print(each)
        each_data =  extract_series(each)
        # standize for the each series:
        each_v = each_data.reshape([-1])
        print("each_v shape:", each_v.shape)
        each_data_mean = np.mean(each_v, axis=0)
        each_data_std = np.std(each_v, axis=0)
        print("each mean:", each_data_mean)
        print("each std:", each_data_std)

        each_sta_data = (each_data - each_data_mean)/each_data_std
        print("standardized data range [{}, {}]:".format(each_sta_data.min(), each_sta_data.max()))

        # if i == 0:
        #     all_data_list = each_data.reshape([-1]).tolist()
        # else:
        #     all_data_list.extend(each_data.reshape([-1]).tolist())
        print("all data list typ:", type(all_data_list))
        # print("each list data len:", len(all_data_list))
        # print("each_data range:[{}, {}]".format(each_data.min(), each_data.max()))
        print("each data shape:", each_data.shape)  # plt histogram of the number of slices
        # plot_hist(each_data.reshape([-1]).tolist(), ax2, x_name="Range of intensity", num_of_bins=50)
        plot_hist(each_sta_data.reshape([-1]).tolist(), ax4, x_name="Range of intensity", num_of_bins=50)
        num_slices_in_training_dataset.append(each_data.shape[-1])

    # plot_hist(num_slices_in_training_dataset, ax, x_name="The number of slices in each case", num_of_bins=50)

    # for each_input_path, each_label_path in zip(train_input_series_paths, train_labels_series_paths):
    #     print("each_input_path:", each_input_path)
    #     print("each_label_path:", each_label_path)
    #     each_input = extract_series(each_input_path)
    #     each_label = extract_series(each_label_path)
    #     print("each_input shape:", each_input.shape)
    #     print("each_label shape:", each_label.shape)
    #     assert  each_input.shape == each_label.shape, "input and label shapes are not matched"
    #
    #     annotated_pixels = each_input[np.where(each_label >0)]
    #     print("annotated pixel range: [{}, {}]".format(annotated_pixels.max(), annotated_pixels.min()))
    #     print("annotated pixel shape:", annotated_pixels.shape)
    #     plot_hist(annotated_pixels.reshape([-1]).tolist(), ax3, x_name="Range of intensity", num_of_bins=50)
    #     label_range_list.append(annotated_pixels.min())
    #     label_range_list.append(annotated_pixels.max())
    ax4.set_xlim([-100, 100])
    # ax2.set_xlim([-5000, 5000])
    # ax3.set_xlim([-5000, 5000])
    # start, end = ax2.get_xlim()
    # print("found annotated pixels range [{}, {}]".format(min(label_range_list), max(label_range_list)))
    # ax2.xaxis.set_ticks(np.arange(start, end, 100))
    plt.show()