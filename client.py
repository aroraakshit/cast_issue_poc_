from __future__ import print_function

import argparse
import time
import numpy as np
import json
import tensorflow as tf
from grpc.beta import implementations

import create_dataset as creat
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import input_pb2 as final_inp

def run(host, port, input_str, model, signature_name):
    
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Pre-processing input
    prediction_input = [json.dumps(eval(input_str))]
    ink, classname = creat.parse_line(prediction_input[0])

    # encapsulate as tf.Example object
    classnames = ['doodle', 'expression', 'symbols']
    features = {}
    features["class_index"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[classnames.index("doodle")]))
    features["ink"] = tf.train.Feature(float_list=tf.train.FloatList(value=ink.flatten()))
    features["shape"] = tf.train.Feature(int64_list=tf.train.Int64List(value=ink.shape))
    f = tf.train.Features(feature=features)
    example = tf.train.Example(features=f)
    final_req = [example]
    start = time.time()
    
    #generate request
    request = classification_pb2.ClassificationRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    request.input.example_list.examples.extend(final_req)
    
    result = stub.Classify(request, 10.0)

    end = time.time()
    time_diff = end - start

    print(result)
    print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Tensorflow server host name', default='0.0.0.0', type=str)
    parser.add_argument('--port', help='Tensorflow server port number', default=9000, type=int)
    parser.add_argument('--input', help='input string', type=str)
    parser.add_argument('--model', help='model name', default="", type=str)
    parser.add_argument('--signature_name', help='Signature name of saved TF model', default='serving_default', type=str)

    args = parser.parse_args()
    run(args.host, args.port, args.input, args.model, args.signature_name)