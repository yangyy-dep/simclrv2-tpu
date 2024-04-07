# ===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ===----------------------------------------------------------------------===#
# -*- coding: utf-8 -*-
import os
import time
import json
import cv2
import numpy as np
import argparse
import glob
import sophon.sail as sail
import logging
import pickle
logging.basicConfig(level=logging.INFO)


class SimCLR2(object):
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.input_shapes = [self.net.get_input_shape(self.graph_name, name) for name in self.input_names]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_shapes = [self.net.get_output_shape(self.graph_name, name) for name in self.output_names]
        logging.debug("load {} success!".format(args.bmodel))
        logging.debug(str(("graph_name: {}, input_names & input_shapes: ".format(self.graph_name), self.input_names,
                           self.input_shapes)))
        logging.debug(str(("graph_name: {}, output_names & output_shapes: ".format(self.graph_name), self.output_names,
                           self.output_shapes)))
        self.input_name = self.input_names[0]
        self.input_shape = self.input_shapes[0]

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

        self.handle = self.net.get_handle()
        self.bmcv = sail.Bmcv(self.handle)
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_names[0])
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)

    def preprocess(self, img):
        img = img.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))  # (10000, 32, 32, 3)

        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose((0, 3, 1, 2))
        return img

    def predict(self, input_img):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)

        return list(outputs.values())[0]

    def postprocess(self, outputs):
        res = list()
        outputs_exp = np.exp(outputs)#(10,1)

        outputs = outputs_exp / np.sum(outputs_exp, axis=1)[:, None]
        predictions = np.argmax(outputs, axis=1)
        for pred, output in zip(predictions, outputs):

            print("Prediction:", pred)


            score = output[pred]
            print("Score:", score)
            res.append((pred.tolist(), float(score)))
        return res

    def __call__(self, img):
        # img_num = len(img_list)
        # img_input_list = []
        # for img in img_list:
        start_time = time.time()
        img = self.preprocess(img)
        self.preprocess_time += time.time() - start_time

        input_img = np.stack(img)
        start_time = time.time()
        outputs = self.predict(input_img)
        self.inference_time += time.time() - start_time
        start_time = time.time()
        res = self.postprocess(outputs)
        self.postprocess_time += time.time() - start_time
        return res

    def get_time(self):
        return self.dt
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def main(args):
    simclr2 = SimCLR2(args)
    batch_size = simclr2.batch_size

    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    results_list = []
    #返回字典
    decode_time = 0.0
    start_time = time.time()
    test_dict = unpickle(args.input)
    decode_time += time.time() - start_time
    testdata = test_dict[b'data'].astype('float')
    testlabels = np.array(test_dict[b'labels'])
    filename = (test_dict[b'filenames'])
    data_len = testdata.shape[0]
    print('数据的长度为')
    print(data_len)
    results_list = list()
    for i in range(0, data_len, batch_size):
        batch_data = testdata[i:i + batch_size]
        results = simclr2(batch_data)
        new_filename = filename[i]
        res_dict = dict()
        logging.info("filename: {}, res: {}".format(new_filename, results[0]))
        print('results_list:',results)
        res_dict['filename'] = new_filename
        res_dict['prediction'] = results[0][0]
        res_dict['score'] = results[0][1]
        results_list.append(res_dict)

    #save result
    if args.input[-1] == '/':
        args.input = args.input[:-1]
    json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_opencv" + "_python_result.json"
    with open(os.path.join(output_dir, json_name), 'w') as jf:
        # json.dump(results_list, jf)
        json.dump(results_list, jf,cls=MyEncoder, indent=4, ensure_ascii=False)
    logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

    # calculate speed
    cn = len(results_list)
    logging.info("------------------ Inference Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = simclr2.preprocess_time / cn
    inference_time = simclr2.inference_time / cn
    postprocess_time = simclr2.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    '''--------------------测试--------------------'''
    print(batch_size)


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default=r'F:\download\SimCLR2\dataset\cifar-10-batches-py\test',
                        help='path of input, must be image directory')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/simclr2_fp32_1b.bmodel',
                        help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argsparser()
    main(args)
