import os
import numpy as np 
import cv2

def load_data(data_path='./data/train/', one_hot=True, test_length=None, img_resize=(256, 256)):
    '''
        This function returns the loaded data(img, label) as in numpy.
    '''
    filename_list = os.listdir(data_path)

    if test_length is not None:
        length = test_length
        filename_list = filename_list[:length]
    else:
        length = len(filename_list)

    np.random.shuffle(filename_list)
    print(len(filename_list))
    img_set = np.empty((length, img_resize[0], img_resize[1], 3), np.float32)
    label_val_set = np.empty((length), np.uint8)

    for idx, filename in enumerate(filename_list):
        print("\r>>> Loading data: {}/{}".format(idx, length), end='')

        img = cv2.imread(os.path.join(data_path, filename))
        img = cv2.resize(img, img_resize).astype(np.float32)

        label = filename.split('.')[0]
        if label == 'cat':
            label_val = 0
        else:
            label_val = 1
        
        img_set[idx] = img
        label_val_set[idx] = label_val
    print('\n')

    if one_hot:
        label_one_hot_set = np.zeros((length, 2), np.uint8)
        label_one_hot_set[np.arange(length), label_val_set] = 1
        label_val_set = label_one_hot_set

    print(label_one_hot_set)

    return img_set, label_val_set 


class Data_set(object):
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        
    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    