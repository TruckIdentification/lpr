import tensorflow as tf
import sys
import os
import cv2
import chepai as cp
dict = {'jing': '京', 'hu': '沪', 'yue': '粤', 'n': 'N','b': 'B','v': 'V', 'c': 'C', 'x': 'X'
    , 'z': 'Z', 'l': 'L', 'k': 'K', 'j': 'J', 'h': 'H', 'g': 'G','m': 'M'
    , 'f': 'F', 'd': 'D', 's': 'S', 'a': 'A', 'p': 'P', 'u': 'U','min': '闽'
    , 'y': 'Y', 't': 'T', 'r': 'R', 'e': 'E', 'w': 'W', 'q': 'Q','su': '苏'
    , '9': '9', '8': '8', '7': '7', '6': '6', '5': '5', '4': '4','zhe': '浙'
    , '3': '3', '2': '2', '1': '1', '0': '0'}
path = 'partition'
dir_path = 'partition/'
image_datas = cv2.imread('111.png')
an = cp.partition(image_datas)
pai_out = ""
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


    # for infile in os.listdir(path):
    #     image_data = tf.gfile.FastGFile(dir_path+infile, 'rb').read()
    # image_dataa = tf.gfile.FastGFile('pr.png', 'rb').read()
    # print(image_dataa)



    with tf.Session() as sess:
        for image_data in an:
            cv2.imwrite('1.png', image_data)
            image_dataa = tf.gfile.FastGFile('1.png', 'rb').read()
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, \
                    {'DecodeJpeg/contents:0': image_dataa})
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            # for node_id in top_k:
            pai = label_lines[top_k[0]]
            score = predictions[0][top_k[0]]
            # print('%s (score = %.5f)' % (pai, score))
            # print('------------------------------------------------------------')
            pai_out+=dict[str(pai)]


print("---------------------------------------------")
print("---------------------------------------------")
print("----------------下面输出车牌-------------------")
print("----------------"+pai_out+"-------------------")
print("---------------------------------------------")
print("---------------------------------------------")
