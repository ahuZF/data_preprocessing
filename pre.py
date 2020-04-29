import os
import numpy as np
import cv2
import sys

class char_cnn_net:
    def __init__(self):
        self.dataset = numbers + alphbets + chinese
        self.dataset_len = len(self.dataset)
        self.img_size = 20
        self.y_size = len(self.dataset)  ### 类别标签长度
        self.batch_size = 100

        self.x_place = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size, self.img_size], name='x_place')
        self.y_place = tf.placeholder(dtype=tf.float32, shape=[None, self.y_size], name='y_place')
        self.keep_place = tf.placeholder(dtype=tf.float32, name='keep_place')

    def list_all_files(self, root):  ###  返回文件路径
        files = []
        list = os.listdir(root)
        for i in range(len(list)):
            element = os.path.join(root, list[i])
            if os.path.isdir(element):
                temp_dir = os.path.split(element)[-1]
                if temp_dir in self.dataset:
                    files.extend(self.list_all_files(element))  ### 在列尾部添加值
            elif os.path.isfile(element):
                files.append(element)
        return files


    def init_data(self, dir):
        X = []
        y = []
        if not os.path.exists(data_dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(dir)

        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            if src_img.ndim == 3:
                continue
            resize_img = cv2.resize(src_img, (20, 20))
            X.append(resize_img)
            # 获取图片文件全目录
            dir = os.path.dirname(file)
            # 获取图片文件上一级目录名
            dir_name = os.path.split(dir)[-1]  ######### 制作y标签
            vector_y = [0 for i in range(len(self.dataset))]  ###制作类别长的标签
            index_y = self.dataset.index(dir_name)  ###找到该文件目录索引
            vector_y[index_y] = 1  ## 该y标签的对于索引赋值
            y.append(vector_y)

        X = np.array(X)
        y = np.array(y).reshape(-1, self.dataset_len)
        return X, y


    def init_testData(self, dir):
        test_X = []
        if not os.path.exists(test_dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(test_dir)
        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            if src_img.ndim == 3:
                continue
            resize_img = cv2.resize(src_img, (20, 20))
            test_X.append(resize_img)
        test_X = np.array(test_X)
        return test_X

if __name__ == '__main__':
    cur_dir = sys.path[0]   ## 其实就是存放需要运行的代码的路径
    data_dir = os.path.join(cur_dir, 'images/images/cnn_char_train')
    test_dir = os.path.join(cur_dir, 'images/images/cnn_char_test')
    train_model_path = os.path.join(cur_dir, 'logs\model.ckpt')
    model_path = os.path.join(cur_dir,'logs\model.ckpt-700')

    train_flag =0
    net = char_cnn_net()

    if train_flag == 1:
        # 训练模型
        net.train(data_dir,train_model_path)
    else:
        # 测试部分
        test_X = net.init_testData(test_dir)
        text = net.test(test_X,model_path)
        print(text)