import os
import cv2
import numpy as np
import struct
def save_mnist_to_jpg(image_data,label_data,output_path,label_path,kind): 

    with open(image_data, 'rb') as f1: 
        magic_number, num_file, height, width = struct.unpack('>IIII',f1.read(16))
        size = height*width
        image_file = f1.read()

    with open(label_data, 'rb') as f2:
        magic_number,num_file = struct.unpack('>II',f2.read(8))       
        label_file = f2.read()

    img_label=open(label_path,'w+')
    for i in range(num_file):
        label = label_file[i]
        image_list = [item for item in image_file[i * size:i * size + size]]
        image_np = np.array(image_list, dtype=np.uint8).reshape(height, width)
        save_name = os.path.join(output_path, '{}_{}_{}.jpg'.format(kind, i, label))
        cv2.imwrite(save_name, image_np)


        label_name=save_name+"   "+str(label)+'\n'
        img_label.write(label_name)
    img_label.close()
    print("=" * 20, "preprocess data finished", "=" * 20)
 
if __name__ == '__main__':
    #压缩的图片文件所在位置
    image_data = '/home/ubuntu/xlj/optimization/Datasets/train-images-idx3-ubyte.gz'
    #压缩的标签文件所在位置
    label_data = '/home/ubuntu/xlj/optimization/Datasets/train-labels-idx1-ubyte.gz'
    output_path= '/home/ubuntu/xlj/optimization/Datasets/train_images/'      
    #标签文件所在位置
    label_path="/home/ubuntu/xlj/optimization/Datasets/train_images/label.txt"  
    #数据集类型用于给图片起名
    kind='train'                                             
    save_mnist_to_jpg(image_data,label_data,output_path,label_path,kind)
