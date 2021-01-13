import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as krs_image
import cv2
import sys
import os
import math
import SimpleITK as sitk

'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''

batch_size =4
ANGLE_STEP  =  1.8888888888888888
NUM_ANGLES  = int(360 // ANGLE_STEP) # 24
print("ANGLE_STEP:", ANGLE_STEP)
rotation_range = 50.0
width_shift_range = 0.16666666666666666
height_shift_range = 0.16666666666666666
zoom_range = 0.16666666666666666
shear_range= 0.19444444444444445
horizontal_flip = True
brightness_range = [0.6684210526315789, 1.131578947368421]
grid_size_multiplier = 4
anchor_mask = [[0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8]] #that should be optimized
plt.rcParams.update({'font.size': 22})
contours_compare_root = "E:\\dataset\\Lung\\check_generator/"


def extract_series(series_path):
    img_obj = nib.load(series_path)
    # print("series:", img_obj.shape)
    print("img.get_data_dtype() == np.dtype(np.int16):", img_obj.get_data_dtype() == np.dtype(np.int16))
    # print("img.affine.shape:", img_obj.affine.shape)
    data = img_obj.get_fdata()
    # print("data in numpy:", data.shape)
    print("data in numpy range: [{}, {}]".format(data.min(), data.max()))
    return data

# def plot_hist(data, axis, x_name,num_of_bins = 1000):
#     axis.hist(data, bins=num_of_bins, color='blue', edgecolor='black', alpha=0.1)
#     axis.set_title("Histogram")
#     axis.set_xlabel(x_name)
#     axis.set_ylabel("Percentage of data")
#     ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(data)))

def encode_polygone(img_path, contours, MAX_VERTICES =1000):
    "give polygons and encode as angle, ditance , probability"
    skipped = 0
    polygons_line = ''
    c = 0
    my_poly_list =[]
    for obj in contours:
        # print(obj.shape)
        myPolygon = obj.reshape([-1, 2])
        # print("mypolygon:", myPolygon.shape)
        if myPolygon.shape[0] > MAX_VERTICES:
            print()
            print("too many polygons")
            break
        my_poly_list.append(myPolygon)

        min_x = sys.maxsize
        max_x = 0
        min_y = sys.maxsize
        max_y = 0
        polygon_line = ''

        # for po
        for x, y in myPolygon:
            # print("({}, {})".format(x, y))
            if x > max_x: max_x = x
            if y > max_y: max_y = y
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            polygon_line += ',{},{}'.format(x, y)
        if max_x - min_x <= 1.0 or max_y - min_y <= 1.0:
            skipped += 1
            continue

        polygons_line += ' {},{},{},{},{}'.format(min_x, min_y, max_x, max_y, c) + polygon_line

    annotation_line = img_path + polygons_line

    return annotation_line, my_poly_list

def My_bilinear_decode_annotationlineNP_inter(encoded_annotationline, MAX_VERTICES=1000, max_boxes=80):
    """
    :param encoded_annotationline: string for lines of img_path and objects c and its contours
    :return: box_data(min_x, min_y, max_x, max_y, c, dists1.dist2...) shape(b, NUM_ANGLE+5)
    """
    # print(COUNT_F)
    # preprocessing of lines from string  ---> very important otherwise can not well split
    annotation_line = encoded_annotationline.split()
    # print(lines[i])
    for element in range(1, len(annotation_line)):
        # print(element)
        for symbol in range(annotation_line[element].count(',') - 4, MAX_VERTICES * 2, 2):
            annotation_line[element] = annotation_line[element] + ',0,0'
    box = np.array([np.array(list(map(float, box.split(','))))
                    for box in annotation_line[1:]])
    # print("box:", box[0])
    # correct boxes
    box_data = np.zeros((max_boxes, 5 + NUM_ANGLES))
    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box), 0:5] = box[:, 0:5]
    # start polygon --->
    # print("len(b):", len(box))
    for b in range(0, len(box)):
        dist_signal = []
        angle_signal = []
        x_old = []
        y_old = []
        boxes_xy = (box[b, 0:2] + box[b, 2:4]) // 2
        # print(boxes_xy)
        # print("b:", b)
        for i in range(5, MAX_VERTICES * 2, 2):

            if box[b, i] == 0 and box[b, i + 1] == 0 and i!=5:
                # print("i:", i)
                # print(box[b, i])
                # print(box[b, i + 1])
                # plt.plt(range(len(box[b]), box[b]))
                # plt.show()
                break
            dist_x = boxes_xy[0] - box[b, i]
            dist_y = boxes_xy[1] - box[b, i + 1]

            dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))

            # if (dist < 1):
            #     print("there is dist< 1")
            #     dist = 1

            angle = np.degrees(np.arctan2(dist_y, dist_x))
            if (angle < 0): angle += 360

            dist_signal.append(dist)
            angle_signal.append(angle)
            # print("dist_signal.len,", len(dist_signal))
            # print("dist_signal.len,", len(angle_signal))
            pair_signal = sorted(zip(angle_signal, dist_signal))
            # for signal in pair_signal:
            #     print(signal)
            # print(pair_signal)
            sorted_angle_signal, sorted_distance =  zip(*pair_signal)
            # print("sorted_angle_signal:", len(sorted_angle_signal))
            # print("sorted_distance:", len(sorted_distance))
            x_old = list(sorted_angle_signal)
            y_old =  list(sorted_distance)
        # print("x_old:", np.array(x_old).shape)
        # print("y_old:", np.array(y_old).shape)

       # use numpy.interp
        x_new = np.linspace(0, 359, NUM_ANGLES, endpoint=False)
        dist_angle_new = np.interp(x_new, x_old,  y_old)
        box_data[b, 5 :] = dist_angle_new

    return box_data

def my_preprocess_true_boxes_NPinterp(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5+69)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    # distace nomralized by diagonal of  box [0.00000001, 1000]
    true_boxes[:,:, 5:NUM_ANGLES + 5] /= np.clip(np.expand_dims(np.sqrt(np.power(boxes_wh[:, :, 0], 2) + np.power(boxes_wh[:, :, 1], 2)), -1), 0.0001, 9999999)

    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: grid_size_multiplier}[l] for l in range(1)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes + NUM_ANGLES),
                       dtype='float32') for l in range(1)]


    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0


    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor): # search for best anchor w,h size in the anchor mask setting
            l = 0
            if n in anchor_mask[l]:
                i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')# for each image b , and each box t
                j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                k = anchor_mask[l].index(n)
                c = true_boxes[b, t, 4].astype('int32')  # class number e.g 10 classses = [0 ,1, 2,3,4,...9], in our case ,one class only , c=0

                y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4] # set as raw  xmin, ymin , xmax ymax
                y_true[l][b, j, i, k, 4] = 1 # F/B label
                y_true[l][b, j, i, k, 5 + c] = 1  # class label as 1 as one-hot for class score. e.g. if 3 class, c =2 , then label in [5: num_classes] into  [0, 0 ,1]
                y_true[l][b, j, i, k, 5 + num_classes:5 + num_classes + NUM_ANGLES] = true_boxes[b, t, 5: 5 + NUM_ANGLES]
    return y_true


def my_get_random_data(input, mask, input_shape, image_datagen, mask_datagen, train_or_test):

    input = np.expand_dims(input,-1)
    input =  np.concatenate([input, input, input], -1)
    # load data ------------------------>
    # image_name = os.path.basename(img_path).replace('.JPG', '')
    # mask_name = os.path.basename(mask_path).replace('.JPG', '')
    # print("img name:", image_name)
    # print("mask name:", mask_name)
    # image = krs_image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    # mask = krs_image.load_img(mask_path, grayscale=True, target_size=(input_shape[0], input_shape[1]))
    input = krs_image.img_to_array(input)
    mask = krs_image.img_to_array(mask)
    # input = np.expand_dims(input, 0)
    # mask = np.expand_dims(mask, 0)
    # print("img shape before aug:", image.shape)
    # print("mask shape before aug:", mask.shape)
    # augment data ----------------------->
    if train_or_test == "Train":
        # print("train aug")
        seed = np.random.randint(0, 2147483647)
        print("aug_seed:", seed)
        print("input shape:", input.shape)
        print("mask shape:", mask.shape)
        aug_image = image_datagen.random_transform(input, seed=seed)

        aug_mask = mask_datagen.random_transform(mask, seed=seed)

        copy_mask = aug_mask.copy().astype(np.uint8)
        print("aug_image_range:[{}, {}]".format(aug_image.min(), aug_image.max()))
        print("copy_mask_range:[{}, {}]".format(copy_mask.min(), copy_mask.max()))
    else:
        # print("Test no aug")
        aug_image = input
        copy_mask = mask.copy().astype(np.uint8)

    # print("mask shape after aug:", np.squeeze(aug_mask).shape)
    # aug_image = krs_image.img_to_array(aug_image)
    # aug_mask = krs_image.img_to_array(aug_mask)
    # find polygons with augmask ------------------------------------>
    # imgray = cv2.cvtColor(np.squeeze(copy_mask), cv2.COLOR_BGR2GRAY)
    # print(copy_mask)
    # ret, thresh = cv2.threshold(copy_mask, 127, 255, 0)  # this require the numpy array has to be the uint8 type
    ret, thresh = cv2.threshold(copy_mask, 0.5, 1, 0)  # this require the numpy array has to be the uint8 type
    aug_mask =thresh
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    selected_coutours = []
    for x in range(len(contours)):
        # print("countour x:", x, contours[x].shape)
        if contours[x].shape[0] > 20:  # require one contour at lest 8 polygons(360/40=9)
            selected_coutours.append(contours[x])
    print("# selected_coutours:", len(selected_coutours))



    # encode contours into annotation lines ---->
    annotation_line, myPolygon = encode_polygone("path", selected_coutours)
    # decode contours annotation line into distance
    box_data = My_bilinear_decode_annotationlineNP_inter(annotation_line)

    # normal the image ----------------->
    # aug_image = aug_image / 255.0
    # aug_mask = aug_mask / 255.0
    aug_mask = aug_mask
    aug_mask =  np.expand_dims(aug_mask, -1)  # since in our case ,we only have one class, if multiple classes binary labels concate at the last dimension
    # print("aug_mask.shape:", aug_mask.shape)
    return aug_image, box_data, myPolygon, aug_mask, annotation_line

def warm_up_func(train_input_series, train_mask_series):
    train_mean_list_series = []
    train_std_list_series = []
    train_nonzeros_indexs_series = []
    train_nonzeros_slices_num = 0
    for each_input_series, each_mask_series in zip(train_input_series, train_mask_series):

        non_zeros_indexes = []

        data_input = load_itk(each_input_series)
        data_mask = load_itk(each_mask_series)
        each_v = data_input.reshape([-1])
        print("each_v shape:", each_v.shape)
        each_data_mean = np.mean(each_v, axis=0)
        each_data_std = np.std(each_v, axis=0)
        train_mean_list_series.append(each_data_mean)
        train_std_list_series.append(each_data_std)
        print("each train series data shape:", data_input.shape)

        assert data_input.shape[-1] == data_mask.shape[-1]
        for i in range(data_mask.shape[-1]):
            temp_mask = data_mask[:, :, i]
            # print("each slice mask shape:", temp_mask.shape)
            nonzeros_in_mask = np.count_nonzero(temp_mask)
            if nonzeros_in_mask > 5:
                # print("find non zeros in the mask at {}: {}".format(i, nonzeros_in_mask))
                non_zeros_indexes.append(i)
                train_nonzeros_slices_num += 1
            else:
                pass
        train_nonzeros_indexs_series.append(non_zeros_indexes)

    return train_mean_list_series, train_std_list_series, train_nonzeros_indexs_series, train_nonzeros_slices_num

def LungGen(inputSeries, labelSeries, input_mean_list_series , input_std_list_series, input_nonzeros_indexs_series, total_samples, batch_size, input_shape, anchors, num_classes, train_flag):
    """
    :param images_list:
    :param masks_list:
    :param batch_size:
    :param input_shape:
    :param train_flag:  STRING Train or else:
    :return:
    """
    case_count =0
    total_num_cases = len(labelSeries)
    print("total_num_cases:", total_num_cases)
    # n = total_num_cases * 256
    # augment generator
    img_data_gen_args = dict(rotation_range=rotation_range,
                             width_shift_range=width_shift_range,
                             height_shift_range=height_shift_range,
                             zoom_range=zoom_range,
                             shear_range=shear_range,
                             horizontal_flip=horizontal_flip,
                             # brightness_range=brightness_range
                             )
    mask_data_gen_args = dict(rotation_range=rotation_range,
                              width_shift_range=width_shift_range,
                              height_shift_range=height_shift_range,
                              zoom_range=zoom_range,
                              shear_range=shear_range,
                              # horizontal_flip=horizontal_flip
                              )
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**mask_data_gen_args)

    ziped_series_mask_list = list(zip(inputSeries, labelSeries,
                                      input_mean_list_series, input_std_list_series,
                                      input_nonzeros_indexs_series))

    total_count = 0
    one_case_conter = 0
    epoch = 0
    while True:
        image_data_list = []
        box_data_list = []
        mask_data_list = []
        # raw_img_path =[]
        # raw_mask_path = []
        # # mypolygon_data = []
        # my_annotation = []
        # print(images_list)
        # print(masks_list)
        if case_count % total_num_cases == 0 and one_case_conter ==0 and total_count ==0 and train_flag == "Train":
            np.random.shuffle(ziped_series_mask_list)
            inputSeries, labelSeries, input_mean_list_series, input_std_list_series, input_nonzeros_indexs_series = zip(*ziped_series_mask_list)  # case shuffle not shuffle inside the case

        temp_input_series = inputSeries[case_count]
        temp_label_series = labelSeries[case_count]
        input_mean_series =  input_mean_list_series[case_count]
        input_std_series = input_std_list_series[case_count]
        # print("input_nonzeros_indexs_series:", input_nonzeros_indexs_series)
        input_nonzeros_indexs = input_nonzeros_indexs_series[case_count]
        # case_count =  (case_count + 1) % total_num_cases

       # extract slices: ---------------->
        case_name = os.path.basename(temp_input_series)
        # print("case name:", case_name)
        input_data = extract_series(temp_input_series)
        label_data = extract_series(temp_label_series)

        if len(input_nonzeros_indexs) != 0:
            print("found num of non zeros slices:", len(input_nonzeros_indexs))
        # select nonzero slices:
        non_zero_inputs = input_data[:,:, input_nonzeros_indexs]
        non_zero_masks = label_data[:, :, input_nonzeros_indexs]
        b = 0
        print("non_zero_inputs shape:", non_zero_inputs.shape)
        while b < batch_size:
            # print("True")
            # if count == 0 and train_flag == "Train":
            #     np.random.shuffle(ziped_img_mask_list)
            # images_list, masks_list = zip(*ziped_img_mask_list)
            # print("------------------------>\n")
            print("total slices in current case:", non_zero_inputs.shape[-1])
            print("current case index:", case_count)
            print("image index in total:", total_count)
            print("image index in current case:", one_case_conter)
            print("image index in a batch:", b)
            temp_img = non_zero_inputs[:, :, one_case_conter]
            temp_mask = non_zero_masks[:, :, one_case_conter]
            # standardize:
            temp_img = (temp_img - input_mean_series)/input_std_series
            # temp_mask = (temp_mask - input_mean_series)/input_std_series
            # plot check temp slice image and label
            # fig = plt.figure(figsize=(8, 8))
            # for i in range(0, 0):
            #     fig.add_subplot(1, 2, i)
            #     plt.imshow(temp_img)
            #     fig.add_subplot(1, 2, i+1)
            #     plt.imshow(temp_mask)
            # plt.show()

            # print("temp_img shape:", temp_img.shape)
            # print("temp_mask shape:", temp_mask.shape)
            # print("temp_img range [{}, {}]:".format(temp_img.min(), temp_img.max()))
            # print("temp_mask range [{}, {}]:".format(temp_mask.min(), temp_mask.max()))

            # if temp_img.min() == temp_img.max() and temp_img.min() ==0:
            # if temp_mask.max() == 0:
            #     one_case_conter = (one_case_conter + 1) % input_data.shape[-1]
            #     total_count = (total_count + 1) % n


            # ----------------->
            img, box, myPolygon, aug_mask, selected_coutours = my_get_random_data(temp_img, temp_mask,
                                                                                  input_shape, image_datagen,
                                                                                  mask_datagen,
                                                                                  train_or_test=train_flag)
            # img, box, aug_mask = my_get_random_data(temp_img, temp_mask, input_shape, image_datagen, mask_datagen,
            #                                         train_or_test=train_flag,
            #                                         one_case_conter=one_case_conter, case_count=case_count,
            #                                         case_name=case_name)
            one_case_conter = (one_case_conter + 1) % non_zero_inputs.shape[-1]
            total_count = (total_count + 1) % total_samples
            b += 1
            # if (one_case_conter + 1) % input_data.shape[-1] == 0:
            #     case_count = (case_count + 1) % total_num_cases

            if one_case_conter % non_zero_inputs.shape[-1] == 0:
                case_count = (case_count + 1) % total_num_cases
            # print("input shape: ", img.shape)
            # print("mask shape:", aug_mask.shape)
            # check the data
            if total_count==total_samples:
                epoch+=1
            background = np.ones(img.shape)*255

            for c in myPolygon:
                # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                cv2.drawContours(background, [c], -1, (0, 255, 0), 2)

                # if aug_mask[cY,cX] ==0:
                #     cv2.circle(background, (cX, cY), 7, (222, 100, 170), -1)
                # else:
                #     cv2.circle(background, (cX, cY), 7, (0, 255, 0), -1)


                cv2.circle(background, (cX, cY), 7, (0, 255, 0), -1)
                cv2.putText(background, "center", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (222, 100, 255), 2)
                # show the image
                # cv2.imshow("Image", background)
                # cv2.waitKey(0)
                cv2.imwrite(contours_compare_root + "batch{}_idx{}_foundMask_".format(epoch, total_count) + 'mask.jpg',
                            aug_mask * 255)
                cv2.imwrite(contours_compare_root + "batch{}_idx{}_found{}C_".format(epoch, total_count, len(myPolygon)) + 'selected_contour.jpg', background)
            # print("selected_coutours:", selected_coutours)
            # print("found  contours#:", len(myPolygon))
            # cv2.imwrite(contours_compare_root + "batch{}_idx{}_foundMask_".format(epoch, total_count) + 'mask.jpg', aug_mask*255)
            # #
            # cv2.drawContours(background, myPolygon, -1, (60, 180, 75))
            # cv2.imwrite(contours_compare_root + "batch{}_idx{}_found{}C_".format(epoch, total_count, len(myPolygon)) + 'selected_contour.jpg', background)
            # cv2.imshow(" ", background)
            # cv2.waitKey()  # show on line need divided 255 save into folder should remove keep in 0 to 255
            # print("myPolygon.shape:", myPolygon.shape)
            # check there is zero: if there is boundry points

            # print("myPolygon.shape:", myPolygon.shape)
            # # check there is zero: if there is boundry points
            #
            # print("count before next:", count)
            # print("range polygon [{}, {}]".format(myPolygon.min(), myPolygon.max()))

            # print(count)
            # if np.any(myPolygon==0) or np.any(myPolygon==aug_image.shape[0]-1) or np.any(myPolygon==aug_image.shape[1]-1):  # roll back.
            #
            #     print("boundary image")
            #     count -=1
            #     b-=1
            #     continue
            # --------------------->
            # print("total_count after next:", total_count)
            image_data_list.append(img)
            # box_data.append(box)
            box_data_list.append(box)
            mask_data_list.append(aug_mask)

        image_batch = np.array(image_data_list)
        box_batch = np.array(box_data_list)
        mask_batch = np.array(mask_data_list)
        # preprocess the bbox into the regression targets
        y_true = my_preprocess_true_boxes_NPinterp(box_batch, input_shape, anchors, num_classes)
        yield [image_batch, *y_true, mask_batch], \
              [np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size)]

def LungGen2(data_list, batch_size, input_shape, anchors, num_classes, train_flag):
    """
    :param images_list:
    :param masks_list:
    :param batch_size:
    :param input_shape:
    :param train_flag:  STRING Train or else:
    :return:
    """
    # case_count =0
    total_num_slices = len(data_list)
    print("total_num_cases:", total_num_slices)
    random_seeds_list = range(total_num_slices)
    # total batches = 0:
    # total_batches =  total_samples//batch_size
    # n = total_num_cases * 256
    # augment generator
    img_data_gen_args = dict(rotation_range=rotation_range,
                             width_shift_range=width_shift_range,
                             height_shift_range=height_shift_range,
                             zoom_range=zoom_range,
                             shear_range=shear_range,
                             horizontal_flip=horizontal_flip,
                             # brightness_range=brightness_range
                             )
    mask_data_gen_args = dict(rotation_range=rotation_range,
                              width_shift_range=width_shift_range,
                              height_shift_range=height_shift_range,
                              zoom_range=zoom_range,
                              shear_range=shear_range,
                              horizontal_flip=horizontal_flip
                              )
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**mask_data_gen_args)
    count = 0

    while True:
        image_data_list = []
        box_data_list = []
        mask_data_list = []

        b = 0
        while  b < batch_size:
            # print("True")
            if count == 0 and train_flag == "Train":
                print("shuffle!")
                np.random.shuffle(data_list)
            # images_list, masks_list = zip(*ziped_img_mask_list)

            temp_dict_data = data_list[count]
            print("temp_dict_data:", temp_dict_data)
            # temp_mask_path = data_list[count]
            # each_input_series + " " + each_mask_series + " " + str(each_data_mean) + " " + str(each_data_std)
            # key, value =  list(temp_dict_data)[0].split(" ")
            splitted_data, nonzero_index  = temp_dict_data
            splitted_data = splitted_data.split(" ")
            # splitted_data, nonzero_index =  list(temp_dict_data.values())[0]
            print("splitted_data:", splitted_data)
            print("nonzero index:", nonzero_index)
            # seeds_for_test=random_seeds_list[count] is for the pre-training premilinary
            img, box, myPolygon, aug_mask, selected_coutours = my_get_random_data2(splitted_data[0], splitted_data[1],  nonzero_index,
                                                                                  float(splitted_data[2]), float(splitted_data[3]),
                                                                                  image_datagen, mask_datagen,
                                                                                  train_or_test=train_flag)



            print("box range [{}, {}]:".format(box.min(), box.max()))

            # checking the polygon
            # background = np.ones(img.shape) * 255
            background = img * 255
            for c in myPolygon:
                # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                cv2.drawContours(background, [c], -1, (0, 255, 0), 2)

                # if aug_mask[cY,cX] ==0:
                #     cv2.circle(background, (cX, cY), 7, (222, 100, 170), -1)
                # else:
                #     cv2.circle(background, (cX, cY), 7, (0, 255, 0), -1)

                cv2.circle(background, (cX, cY), 7, (0, 255, 0), -1)
                cv2.putText(background, "center", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (222, 100, 255), 2)
                # show the image
                # cv2.imshow("Image", background)
                # cv2.waitKey(0)
                cv2.imwrite(contours_compare_root + "batch{}_idx{}_foundMask_".format(count, b) + 'mask.jpg',
                            aug_mask * 255)
                cv2.imwrite(contours_compare_root + "batch{}_idx{}_found{}C_".format(count, b, len(
                    myPolygon)) + 'selected_contour.jpg', background)


            count = (count + 1) % total_num_slices
            b+=1

            image_data_list.append(img)
            # box_data.append(box)
            box_data_list.append(box)
            mask_data_list.append(aug_mask)
            print("count after next:", count)
        print("data batch image", len(image_data_list))
        print("data batch box", len(box_data_list))
        print("data batch mask", len(mask_data_list))
        image_batch = np.array(image_data_list)
        box_batch = np.array(box_data_list)
        mask_batch = np.array(mask_data_list)
        # preprocess the bbox into the regression targets
        y_true = my_preprocess_true_boxes_NPinterp(box_batch, input_shape, anchors, num_classes)
        # yield [image_batch, *y_true, mask_batch], \
        #       [np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size)]
        yield [image_batch, *y_true, mask_batch], \
              [np.zeros(len(image_data_list)), np.zeros(len(image_data_list)), np.zeros(len(image_data_list)), np.zeros(len(image_data_list)),
               np.zeros(len(image_data_list))]

def my_get_random_data2(input_path, mask_path, nonzero_index, input_mean_list_series, input_std_list_series,
                       image_datagen, mask_datagen, train_or_test):
    # read data
    input_data, origin, spacing = load_itk(input_path)
    mask_data, _, _ = load_itk(mask_path)
    input = input_data[nonzero_index, :, :]
    mask = mask_data[nonzero_index, :, :]
    input = np.expand_dims(input, -1)

    # load data ------------------------>
    # image_name = os.path.basename(img_path).replace('.JPG', '')
    # mask_name = os.path.basename(mask_path).replace('.JPG', '')
    # print("img name:", image_name)
    # print("mask name:", mask_name)
    # image = krs_image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    # mask = krs_image.load_img(mask_path, grayscale=True, target_size=(input_shape[0], input_shape[1]))
    input = krs_image.img_to_array(input)
    mask = krs_image.img_to_array(mask)
    # input = np.expand_dims(input, 0)
    # mask = np.expand_dims(mask, 0)
    # print("img shape before aug:", image.shape)
    # print("mask shape before aug:", mask.shape)
    # augment data ----------------------->
    if train_or_test == "Train":
        # print("train aug")
        seed = np.random.randint(0, 2147483647)
        print("aug_seed:", seed)
        print("input shape:", input.shape)
        print("mask shape:", mask.shape)
        aug_image = image_datagen.random_transform(input, seed=seed)

        aug_mask = mask_datagen.random_transform(mask, seed=seed)

        copy_mask = aug_mask.copy().astype(np.uint8)
        print("aug_image_range:[{}, {}]".format(aug_image.min(), aug_image.max()))
        print("copy_mask_range:[{}, {}]".format(copy_mask.min(), copy_mask.max()))
    else:
        # print("Test no aug")
        aug_image = input
        copy_mask = mask.copy().astype(np.uint8)

    # print("mask shape after aug:", np.squeeze(aug_mask).shape)
    # aug_image = krs_image.img_to_array(aug_image)
    # aug_mask = krs_image.img_to_array(aug_mask)
    # find polygons with augmask ------------------------------------>
    # imgray = cv2.cvtColor(np.squeeze(copy_mask), cv2.COLOR_BGR2GRAY)
    # print(copy_mask)
    # ret, thresh = cv2.threshold(copy_mask, 127, 255, 0)  # this require the numpy array has to be the uint8 type
    ret, thresh = cv2.threshold(copy_mask, 0.5, 1, 0)  # this require the numpy array has to be the uint8 type
    aug_mask =thresh
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    selected_coutours = []
    for x in range(len(contours)):
        # print("countour x:", x, contours[x].shape)
        if contours[x].shape[0] > 20:  # require one contour at lest 8 polygons(360/40=9)
            selected_coutours.append(contours[x])
    # print("# selected_coutours:", len(selected_coutours))


    full_name = input_path+str(nonzero_index)
    # encode contours into annotation lines ---->
    annotation_line, myPolygon = encode_polygone(full_name, selected_coutours)
    # decode contours annotation line into distance
    box_data = My_bilinear_decode_annotationlineNP_inter(annotation_line)

    # normal the image ----------------->
    # aug_image = aug_image / 255.0
    # aug_mask = aug_mask / 255.0


    # # normalization
    # # win1  : window [-3000, 1700]
    # LB1, UB1 = -3000, 1700
    # input1= normalize(aug_image,LB1,UB1)
    # print("input1 range: [{}, {}]".format(input1.min(), input1.max()))
    # # win2  : window  [-1400 500]
    # LB2, UB2 = -1400, 500
    # input2 = normalize(aug_image, LB2, UB2)

    # print("input2 range: [{}, {}]".format(input2.min(), input2.max()))
    # std :
    # standardize:
    inputstd = (aug_image - input_mean_list_series) / input_std_list_series

    final_input =  np.concatenate([inputstd, inputstd, inputstd], axis=-1)
    print("inputstd range: [{}, {}]".format(inputstd.min(), inputstd.max()))
    # final input with 3 channels
    aug_mask = aug_mask
    print("aug_mask range: [{}, {}]".format(aug_mask.min(), aug_mask.max()))
    aug_mask =  np.expand_dims(aug_mask, -1)  # since in our case ,we only have one class, if multiple classes binary labels concate at the last dimension
    # print("aug_mask.shape:", aug_mask.shape)
    return final_input, box_data, myPolygon, aug_mask, annotation_line

def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def plot_ct_scan(scan, num_column=4, jump=1):
    num_slices = len(scan)
    num_row = (num_slices//jump + num_column - 1) // num_column
    f, plots = plt.subplots(num_row, num_column, figsize=(num_column*4, num_row*4))
    for i in range(0, num_row*num_column):
        plot = plots[i % num_column] if num_row == 1 else plots[i // num_column, i % num_column]
        plot.axis('off')
        if i < num_slices//jump:
            plot.imshow(scan[i*jump], cmap="gray")
    plt.tight_layout()

def warm_up_funcV2(train_input_series, train_mask_series):
    # train_mean_list_series = []
    # train_std_list_series = []
    # train_nonzeros_indexs_series = []
    # train_nonzeros_slices_num = 0
    # total_num_batchse=0
    Key_list  = []
    nonzero_list = []
    # data_dict = {}
    data_list = []
    for each_input_series, each_mask_series in zip(train_input_series, train_mask_series):

        non_zeros_indexes = []
        # ct_scan, origin, spacing
        data_input, origin, spacing = load_itk(each_input_series)
        # print("ct_scan:", data_input)
        print("origin:", origin)
        print("spacing:", spacing)
        print("data_input shap:", data_input.shape)
        print("data input range [{}, {}]".format(data_input.min(), data_input.max()))
        data_mask, _, _ = load_itk(each_mask_series)
        print("data mask range [{}, {}]".format(data_mask.min(), data_mask.max()))
        each_v = data_input.reshape([-1])
        print("each_v shape:", each_v.shape)
        each_data_mean = np.mean(each_v, axis=0)
        each_data_std = np.std(each_v, axis=0)
        # train_mean_list_series.append(each_data_mean)
        # train_std_list_series.append(each_data_std)
        print("each train series data shape:", data_input.shape)

        assert data_input.shape[0] == data_mask.shape[0]
        for i in range(data_mask.shape[0]):

            temp_mask = data_mask[i, :, :]
            # print("each slice mask shape:", temp_mask.shape)
            nonzeros_in_mask = np.count_nonzero(temp_mask)
            if nonzeros_in_mask > 5:

                # print("find non zeros in the mask at {}: {}".format(i, nonzeros_in_mask))
                # non_zeros_indexes.append(i)
                # train_nonzeros_slices_num += 1
                key = each_input_series + " " + each_mask_series + " " + str(each_data_mean) + " " + str(each_data_std)
                # value =  i
                Key_list.append(key)
                nonzero_list.append(i)
                # data_input = (data_input - each_data_mean) / each_data_std
                #
                # plot_ct_scan(data_input, num_column=4, jump=1)
                # plot_ct_scan(data_mask, num_column=4, jump=1)

                plt.show()
            else:
                pass
    ziped_list =  zip(Key_list, nonzero_list)

        # total_num_batchse += math.ceil(len(non_zeros_indexes)/ batch_size)
        # train_nonzeros_indexs_series.append(non_zeros_indexes)

    # CALCULATE total number of batches for each
    return list(ziped_list)


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


if __name__ == '__main__':
    bs =4
    current_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    classes_path = current_file_dir_path + '/yolo_classesTongue.txt'
    anchors_path = current_file_dir_path + '/yolo_anchorsTongue.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    train_input_series_paths = []
    train_labels_series_paths = []
    val_input_series_paths = []
    val_labels_series_paths = []
    train_total_series = glob('E:\\dataset\\Prostate\\myDownload\\trainData/*.mhd')
    # val_total_series = glob('E:\\dataset\\Prostate\\myDownload\\trainData/*')

    total_input_series_paths = []
    total_labels_series_paths = []
    for each_train_path in train_total_series:  # find input and label
        # print(each_train_path)
        if "seg" in each_train_path:
            total_labels_series_paths.append(each_train_path)
        else:
            total_input_series_paths.append(each_train_path)

    print("total len of total input series:", len(total_input_series_paths))
    print("total len of total mask series:", len(total_labels_series_paths))

    # warm up ~~~---------------->
    # split train dataset:
    print("splitting the training dataset into train and val")
    train_rate = 0.8

    val_rate = 0.5  # half of the remainning
    # print("raw total series paths:", train_input_series_paths)
    num_train_cases = int(train_rate * len(total_input_series_paths))
    print("splitted num_train_cases:", num_train_cases)

    # num_val_cases = int(val_rate * len(train_input_series_paths))
    # print("splitted num_train_cases:", num_train_cases)

    # for train ->
    # train_input_series = total_input_series_paths[:num_train_cases]
    # print("splitted train_input_series # :", len(train_input_series))
    # train_mask_series = total_labels_series_paths[:num_train_cases]

    train_input_series = total_input_series_paths[0:1]
    print("splitted train_input_series # :", len(train_input_series))
    train_mask_series = total_labels_series_paths[0:1]
    # for val ->
    # val_input_series = total_input_series_paths[
    #                    num_train_cases:num_train_cases+1]   # for check purpose only
    # val_mask_series = total_labels_series_paths[num_train_cases:num_train_cases+1] # for check purpose only
    remaining_input_series = total_input_series_paths[
                             num_train_cases:]  ## split training data into real train and validation
    remaining_mask_series = total_labels_series_paths[num_train_cases:]

    num_val_cases = int(val_rate * len(remaining_input_series))
    val_input_series = remaining_input_series[:num_val_cases]
    print("splitted val_input_series # :", len(val_input_series))
    val_mask_series = remaining_mask_series[:num_val_cases]
    # # for test ->
    # test_input_series = remaining_input_series[num_val_cases:]
    # print("splitted test_input_series # :", len(test_input_series))
    # test_mask_series = remaining_mask_series[num_val_cases:]

    # # for the laptop:
    val_input_series = remaining_input_series[0:1]
    print("splitted val_input_series # :", len(train_input_series))
    val_mask_series = remaining_mask_series[0:1]

    # # for test ->
    test_input_series = remaining_input_series[num_val_cases:]
    # print("splitted test_input_series # :", len(test_input_series))
    test_mask_series = remaining_mask_series[num_val_cases:]

    # anaylize the training CTs
    print("train series len:", len(train_input_series))
    print("val series len:", len(val_input_series))
    print("test series len:", len(test_input_series))
    # warm up for training needed

    print("train warm up ---------------------------->")
    # for each_input_series, each_mask_series in zip(train_input_series, train_mask_series):
    train_data_list = \
        warm_up_funcV2(train_input_series, train_mask_series)

    val_data_list = \
        warm_up_funcV2(val_input_series, val_mask_series)

    print("found nonzero slice:", train_data_list)
    print("found train nonzero slices:", len(train_data_list))
    print("found val nonzero slices:", len(val_data_list))

    # # num_train = train_num_batches
    # # # num_val = val_num_batches
    # print("found nonzero train batches:", train_num_batches)
    # print("found nonzero val batches:", val_num_batches)
    #
    #
    # # create data_generator
    #
    # # for train:
    train_Gen = LungGen2(train_data_list,
                        batch_size=bs,
                        input_shape=(512, 512),
                        anchors=anchors, num_classes=num_classes,
                        train_flag="Train")
    val_Gen = LungGen2(val_data_list,
                      batch_size=bs,
                      input_shape=(512, 512),
                      anchors=anchors, num_classes=num_classes,
                      train_flag="test")
    num_train = int(len(train_data_list))
    num_val = len(val_data_list)

    batch_size = 4  # decrease/increase batch size according to your memory of your GPU
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

for data in val_Gen:
    print(data)