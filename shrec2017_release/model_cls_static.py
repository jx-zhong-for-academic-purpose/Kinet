"""
    Compared with model_baseline, do not use correlation output for skip link
    Compared to model_baseline_fixed, added return values to test whether nsample is set reasonably.
"""

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/sampling'))
from net_utils import *

def placeholder_inputs(batch_size, num_point, num_frames, input_dim):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * num_frames, input_dim))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, num_frames, is_training, bn_decay=None, CLS_COUNT=-1):
    """ Input:
            point_cloud: [batch_size, num_point * num_frames, 3]
        Output:
            net: [batch_size, num_class] """
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value // num_frames
    l0_xyz = point_cloud[:, :, 0:3]
    l0_time = tf.concat([tf.ones([batch_size, num_point, 1]) * i for i in range(num_frames)], axis=-2)
    l0_points = None #l0_time

    RADIUS1 = np.linspace(0.5, 0.6, num_frames, dtype='float32')
    delta_t = 2
    RADIUS1[delta_t:] = 0
    RADIUS2 = RADIUS1 * 2
    RADIUS3 = RADIUS1 * 4

    print("**********************l0_xyz:", l0_xyz.shape)
    l1_xyz, l1_time, l1_points, l1_indices = static_module(l0_xyz, l0_time, l0_points, npoint=32 * 32, radius=RADIUS1, nsample=64, mlp=[64, 128], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    print("**********************l1_xyz:", l1_xyz.shape)
    l2_xyz, l2_time, l2_points, l2_indices = static_module(l1_xyz, l1_time, l1_points, npoint=32 * 16, radius=RADIUS2, nsample=64, mlp=[128, 256], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    print("**********************l2_xyz:", l2_xyz.shape)
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[512, 1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    # Fully connected layers
    net = tf.reshape(l4_points, [batch_size, -1])
    net = tf_util.fully_connected(net, CLS_COUNT, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024*2,6))
        outputs = get_model(inputs, 2, tf.constant(True))
        print(outputs)
