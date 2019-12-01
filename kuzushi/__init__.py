import os
import numpy as np
import cv2
def lload(filename,list):
    cun=[]
    for i in range(len(list)):
        pa=os.path.join(filename,list[i])
        img=cv2.imread(pa)
        img=cv2.resize(img,(512,512))
        cun.append(img)
    return np.array(cun)

class DataSampler(object):
    def __init__(self):
        self.shape = [512, 512, 3]
        self.name = "kuzushi"
        self.db_path = "/home/usr8/n70208b/wgan/all_"
        self.db_files = os.listdir(self.db_path)
        self.cur_batch_ptr = 0
        self.cur_batch = self.load_new_data()
        self.train_batch_ptr = 0
        self.train_size = len(self.db_files) * 10000
        self.test_size = self.train_size
    def load_new_data(self):
        filename = os.path.join(self.db_path, self.db_files[self.cur_batch_ptr])
        self.cur_batch_ptr += 1
        if self.cur_batch_ptr == len(self.db_files):
            self.cur_batch_ptr = 0
        return (lload(self.db_path,self.db_files) -127.5)/127.5
    def lload1(filename):
        img=cv2.imread(filename)
        img=cv2.resize(img,(1000,1500))
        return img
    def __call__(self, batch_size):
        prev_batch_ptr = self.train_batch_ptr
        self.train_batch_ptr += batch_size
        if self.train_batch_ptr > self.cur_batch.shape[0]:
            self.train_batch_ptr = batch_size
            prev_batch_ptr = 0
            self.cur_batch = self.load_new_data()
        x = self.cur_batch[prev_batch_ptr:self.train_batch_ptr, :, :, :]
        return np.reshape(x, [batch_size, -1])
    def data2img(self, data):
        rescaled = np.divide(data + 1.0, 2.0)
        return np.reshape(np.clip(rescaled, 0.0, 1.0), [data.shape[0]] + self.shape)


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])
