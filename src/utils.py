import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
from skimage import io, filters, measure
from scipy import ndimage
import csv

def gen_mask(ins_img):
    mask = []
    for i, mask_i in enumerate(ins_img):
        binarized = mask_i * (i + 1)
        mask.append(binarized)
    mask = np.sum(np.stack(mask, axis=0), axis=0).astype(np.uint8)
    return mask


def coloring(mask):
    ins_color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    n_ins = len(np.unique(mask)) - 1
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_ins)]
    for i in range(n_ins):
        ins_color_img[mask == i + 1] =\
            (np.array(colors[i][:3]) * 255).astype(np.uint8)
    no_organs = report_no_of_organs(ins_color_img)        
    return ins_color_img, no_organs

def report_no_of_organs(img):

    # # Connected component implementation
    # #calling connectedComponentswithStats to get the size of each component
    # grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # nb_comp,output,sizes,centroids=cv2.connectedComponentsWithStats(grayimg,connectivity=4)
    # #taking away the background
    # nb_comp-=1; sizes=sizes[1:,-1]; centroids=centroids[1:,:]
    # print(nb_comp)

    val = filters.threshold_otsu(img)
    drops = ndimage.binary_fill_holes(img < val)
    labels = measure.label(drops)
    print(labels.max())
    return labels.max()
    




# def gen_instance_mask(sem_pred, ins_pred, n_obj):
#     embeddings = ins_pred[:, sem_pred].transpose(1, 0)
#     clustering = KMeans(n_obj).fit(embeddings)
#     labels = clustering.labels_

#     instance_mask = np.zeros_like(sem_pred, dtype=np.uint8)
#     for i in range(n_obj):
#         lbl = np.zeros_like(labels, dtype=np.uint8)
#         lbl[labels == i] = i + 1
#         instance_mask[sem_pred] += lbl

#     return instance_mask

def gen_instance_mask(sem_pred, ins_pred):
    embeddings = ins_pred[:, sem_pred].transpose(1, 0)
    clustering = KMeans(3).fit(embeddings)
    labels = clustering.labels_

    instance_mask = np.zeros_like(sem_pred, dtype=np.uint8)
    for i in range(3):
        lbl = np.zeros_like(labels, dtype=np.uint8)
        lbl[labels == i] = i + 1
        instance_mask[sem_pred] += lbl

    return instance_mask    


# def gen_color_img(sem_pred, ins_pred, n_obj):
#     return coloring(gen_instance_mask(sem_pred, ins_pred, n_obj))

def gen_color_img(sem_pred, ins_pred):
    return coloring(gen_instance_mask(sem_pred, ins_pred))


# MSE loss function
def mse_loss(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    squared_error = (y_pred - y_true) ** 2
    sum_squared_error = np.sum(squared_error)
    loss = sum_squared_error / y_true.size
    rmse =  np.sqrt(loss)
    return loss, rmse


# MAE loss function
def mae_loss(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    abs_error = np.abs(y_pred - y_true)
    sum_abs_error = np.sum(abs_error)
    loss = sum_abs_error / y_true.size
    return loss

# truth = []
# with open("Groundtruth.txt") as f:
#     reader = csv.reader(f)
#     mydict_t = {rows[0]:rows[1] for rows in reader}

# pred = []
# with open("Predicted.txt") as f:
#     reader = csv.reader(f)
#     mydict_p = {rows[0]:rows[1] for rows in reader}

# for key, value in mydict_t.items():
#     truth.append(value)
#     pred.append(mydict_p[key])

# print(mse_loss(pred,truth))
# print(mae_loss(pred,truth))
