#!/usr/bin/env python3
# python3 inpaint.py --allTest 0 --allImagesPath imgs/flowers/ --testImagesPath imgs/flowers/
# python3 inpaint.py --allTest 1

import tensorflow as tf
import numpy as np
import numpy.random as npr
import argparse
import os
import utils as ut
import pdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='saved')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allTest', type=int, default=1)
    parser.add_argument('--numAllTestImgs', type=int, default=2)
    parser.add_argument('--numTrainImgs', type=int, default=4)
    parser.add_argument('--allImagesPath', type=str, default='/')
    parser.add_argument('--testImagesPath', type=str, default='/')

    args = parser.parse_args()

    npr.seed(args.seed)

    save = os.path.expanduser(args.save)

    numTrain = args.numTrainImgs
    numTestImgs = args.numAllTestImgs

    # pdb.set_trace()
    print(numTrain)
    print(numTestImgs)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        # export_dir = "{}/model/anila_celebA".format(args.save)
        export_dir = "model/anila_celebA"
        # export_dir = "model/anila_flowers"
        model = tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], export_dir)
        loaded_graph = tf.compat.v1.get_default_graph()

        # pdb.set_trace()

        inputTensorXName = model.signature_def['predict_images'].inputs['x'].name # 
        # inputTensorXName = 'x:0'
        inputTensorX = loaded_graph.get_tensor_by_name(inputTensorXName)
        inputTensory0Name = model.signature_def['predict_images'].inputs['y0'].name # 'y:0'
        # inputTensory0Name = 'y:0'
        inputTensory0 = loaded_graph.get_tensor_by_name(inputTensory0Name)
        outputTensorName = model.signature_def['predict_images'].outputs['yn'].name # 'add_29:0'
        # outputTensorName = 'add_29:0'
        outputTensor = loaded_graph.get_tensor_by_name(outputTensorName)

        I = npr.randint(numTrain, size=10000)
        _, newTrainY = ut.createBatch(I, path=args.allImagesPath)
        meanY = np.mean(newTrainY, axis=0)

        if args.allTest == 1:
            I = npr.randint(low=numTrain+1, high=numTrain+numTestImgs, size=numTestImgs)
            valXBatch, valYBatch = ut.createBatch(I, path=args.allImagesPath)
        else:
            valXBatch, valYBatch, numTestImgs = ut.createBatchSpec(path=args.testImagesPath)
        y0 = np.expand_dims(meanY, axis=0).repeat(numTestImgs, axis=0)

        resImg = sess.run(outputTensor, {inputTensorX: valXBatch, inputTensory0: y0})
        cw = 10
        if numTestImgs < 10:
            cw = numTestImgs % 10
        ut.saveImgs(valXBatch, resImg, valYBatch, "{}/results/flowers".format(args.save), colWidth=cw)


if __name__=='__main__':
    main()
