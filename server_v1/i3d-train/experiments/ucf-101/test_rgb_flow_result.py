# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import sys
sys.path.append('./i3d-train')
import time
import numpy
import json
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_test
import math
import numpy as np
from i3d import InceptionI3d
from utils import *

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 16, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 27, 'The num of class')
FLAGS = flags.FLAGS

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_training(file):
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.
    global sess
    global norm_score
    global rgb_images_placeholder
    global flow_images_placeholder
    global labels_placeholder
    global is_training

    #all_steps = num_test_videos
    #top1_list = []
    #for step in xrange(all_steps):
        #start_time = time.time()
    s_index = 0
    predicts = []
    top1 = False
    while True:
        print(file)
        rgb_images, flow_images, val_labels, s_index, is_end = input_test.read_clip_and_label(
                        filename=file,
                        batch_size=FLAGS.batch_size * gpu_num,
                        s_index=s_index,
                        num_frames_per_clip=FLAGS.num_frame_per_clib,
                        crop_size=FLAGS.crop_size,
                        )
        predict = sess.run(norm_score,
                           feed_dict={
                                        rgb_images_placeholder: rgb_images,
                                        flow_images_placeholder: flow_images,
                                        labels_placeholder: val_labels,
                                        is_training: False
                                        })
        predicts.append(np.array(predict).astype(np.float32).reshape(FLAGS.classics))
        ans=""
        if is_end:
            avg_pre = np.mean(predicts, axis=0).tolist()
            pre_label_num=avg_pre.index(max(avg_pre))
            f_label=open("./i3d-train/list/ucf_list/label_class27.json","r")
            label_json=json.loads(f_label.read())
            for label in label_json:
                if label["num"]==pre_label_num:
                    print(label["name"])
                    ans=label["name"]
        #avg_pre = np.mean(predicts, axis=0).tolist()
            top1 = (avg_pre.index(max(avg_pre))==val_labels)
            print(avg_pre)
            return ans
            break
    duration = time.time() - start_time
    #print('TOP_1_ACC in test: %f , time use: %.3f' % (top1, duration))
    #print(len(top1_list))
    #print('TOP_1_ACC in test: %f' % np.mean(top1_list))
    #print("done")
    
def load_model():
    global sess
    global norm_score
    global rgb_images_placeholder
    global flow_images_placeholder
    global labels_placeholder
    global is_training
    rgb_pre_model_save_dir = "./i3d-train/experiments/ucf-101/models/rgb_model"
    flow_pre_model_save_dir = "./i3d-train/experiments/ucf-101/models/flow_model"
    with tf.Graph().as_default():
        rgb_images_placeholder, flow_images_placeholder, labels_placeholder, is_training = placeholder_inputs(
                        FLAGS.batch_size * gpu_num,
                        FLAGS.num_frame_per_clib,
                        FLAGS.crop_size,
                        FLAGS.rgb_channels
                        )

        with tf.variable_scope('RGB'):
            rgb_logit, _ = InceptionI3d(
                                num_classes=FLAGS.classics,
                                spatial_squeeze=True,
                                final_endpoint='Logits',
                                name='inception_i3d'
                                )(rgb_images_placeholder, is_training)
        with tf.variable_scope('Flow'):
            flow_logit, _ = InceptionI3d(
                                num_classes=FLAGS.classics,
                                spatial_squeeze=True,
                                final_endpoint='Logits',
                                name='inception_i3d'
                                )(flow_images_placeholder, is_training)
        norm_score = tf.nn.softmax(tf.add(rgb_logit, flow_logit))

        # Create a saver for writing training checkpoints.
        rgb_variable_map = {}
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB' and 'Adam' not in variable.name.split('/')[-1] :
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow'and 'Adam' not in variable.name.split('/')[-1] :
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)

    # load pre_train models
    ckpt = tf.train.get_checkpoint_state(rgb_pre_model_save_dir)
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        rgb_saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")
    ckpt = tf.train.get_checkpoint_state(flow_pre_model_save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        flow_saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")
    

def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()

