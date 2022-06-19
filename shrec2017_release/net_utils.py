""" PointNet++ Layers

Author: Charles R. Qi
Modified by Xingyu Liu
Date: November 2019

Modified by Jiaxing Zhong
Date: November 2021
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(ROOT_DIR))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, query_ball_point_var_rad, query_ball_point_var_rad_var_seed, group_point, knn_point, select_top_k
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    sample_idx = farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, sample_idx) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, sample_idx, grouped_xyz

def sample_and_group_by_index(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True, sample_idx=None):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    if sample_idx is None:
        sample_idx = farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, sample_idx) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, sample_idx, grouped_xyz

def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False, freeze_bn=False, sample_idx=None, trainable=True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        elif sample_idx is not None:
            new_xyz, new_points, idx, sample_idx, grouped_xyz = sample_and_group_by_index(npoint, radius, nsample, xyz, points, knn, use_xyz, sample_idx)
        else:
            new_xyz, new_points, idx, sample_idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training, trainable=trainable,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format, freeze_bn=freeze_bn)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keepdims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keepdims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keepdims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keepdims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keepdims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training, trainable=trainable,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format, freeze_bn=freeze_bn)
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        if sample_idx is not None:
            return new_xyz, new_points, (idx, sample_idx)
        else:
            return new_xyz, new_points, idx


def static_module(xyz, time, points, npoint, radius, nsample, mlp, mlp2, group_all,
                  is_training, bn_decay, scope, module_type='ind_without_time', fps=True, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    '''
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            time: (batch_size, ndataset, 1) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radiuses in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            module_type: 'ind' or 'rel' -- the type of meteor module
            fps: whether to do farthest point sampling; Requires npoint == xyz.get_shape()[1].value, when fps=False
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    assert(module_type is 'ind_without_time') # The time is only used for sampling points.
    data_format = 'NCHW' if use_nchw else 'NHWC'
    sample_idx = None
    batch_size = xyz.get_shape()[0].value
    with tf.variable_scope(scope) as sc:

        if fps:
            ##### sample and group with variable radius
            sample_idx = farthest_point_sample(npoint, xyz)
        else:
            ##### no sampling at all
            sample_idx = tf.tile(tf.expand_dims(tf.range(npoint, dtype=tf.int32), 0), [batch_size, 1])

        new_xyz = gather_point(xyz, sample_idx)  # (batch_size, npoint, 3)
        new_time = gather_point(time, sample_idx)  # (batch_size, npoint, 1)
        time_ = tf.reshape(time, [batch_size, 1, -1])  # (batch_size, 1, ndataset)
        new_time_ = tf.abs(new_time - time_)  # (batch_size, npoint, ndataset)
        radius_ = tf.gather(radius, tf.cast(new_time_, tf.int32))  # (batch_size, npoint, ndataset)
        idx, pts_cnt = query_ball_point_var_rad(radius_, nsample, xyz, new_xyz)

        grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization        
        FRAME_CNT = 32
        grouped_time = group_point(time, idx)  # (batch_size, npoint, nsample * 3, channel)
        grouped_time -= tf.tile(tf.expand_dims(new_time, 2), [1, 1, nsample, 1])  # time-shift normalization
        grouped_time += FRAME_CNT

        if points is not None:
            new_points = gather_point(points, sample_idx)  # (batch_size, npoint, channel)
            grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
            grouped_time = group_point(time, idx) # (batch_size, npoint, nsample, channel)
            if use_xyz:
                if module_type == 'ind':
                    new_points = tf.concat([grouped_xyz, grouped_time, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+1+channel)
                elif module_type == 'ind_without_time':
                    new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+1+channel)
                else:
                    new_points_expand = tf.tile(tf.expand_dims(new_points, 2), [1,1,nsample,1])
                    new_points = tf.concat([grouped_xyz, grouped_time, grouped_points, new_points_expand], axis=-1) # (batch_size, npoint, nample, 3+1+channel+channel)
            else:
                new_points = grouped_points
        else:
            new_points = grouped_xyz

        # Point Feature Embedding
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
                                        

        new_points = tf.reduce_max(new_points, axis=[2], name='maxpool')
        return new_xyz, new_time, new_points, idx


def get_4D_inversion(AtA, group_size, batch_size, npoint, flow_dim, EPS):
    assert(group_size + 1 == 4)
    AtA = tf.reshape(AtA, [batch_size, npoint, flow_dim // group_size, -1])
    AtA0, AtA1, AtA2, AtA3, \
    AtA4, AtA5, AtA6, AtA7, \
    AtA8, AtA9, AtA10, AtA11, \
    AtA12, AtA13, AtA14, AtA15 = tf.unstack(AtA, axis=3)

    inv0 = AtA5 * AtA10 * AtA15 - \
           AtA5 * AtA11 * AtA14 - \
           AtA9 * AtA6 * AtA15 + \
           AtA9 * AtA7 * AtA14 + \
           AtA13 * AtA6 * AtA11 - \
           AtA13 * AtA7 * AtA10
    inv4 = -AtA4 * AtA10 * AtA15 + \
           AtA4 * AtA11 * AtA14 + \
           AtA8 * AtA6 * AtA15 - \
           AtA8 * AtA7 * AtA14 - \
           AtA12 * AtA6 * AtA11 + \
           AtA12 * AtA7 * AtA10
    inv8 = AtA4 * AtA9 * AtA15 - \
           AtA4 * AtA11 * AtA13 - \
           AtA8 * AtA5 * AtA15 + \
           AtA8 * AtA7 * AtA13 + \
           AtA12 * AtA5 * AtA11 - \
           AtA12 * AtA7 * AtA9
    inv12 = -AtA4 * AtA9 * AtA14 + \
            AtA4 * AtA10 * AtA13 + \
            AtA8 * AtA5 * AtA14 - \
            AtA8 * AtA6 * AtA13 - \
            AtA12 * AtA5 * AtA10 + \
            AtA12 * AtA6 * AtA9
    inv1 = -AtA1 * AtA10 * AtA15 + \
           AtA1 * AtA11 * AtA14 + \
           AtA9 * AtA2 * AtA15 - \
           AtA9 * AtA3 * AtA14 - \
           AtA13 * AtA2 * AtA11 + \
           AtA13 * AtA3 * AtA10
    inv5 = AtA0 * AtA10 * AtA15 - \
           AtA0 * AtA11 * AtA14 - \
           AtA8 * AtA2 * AtA15 + \
           AtA8 * AtA3 * AtA14 + \
           AtA12 * AtA2 * AtA11 - \
           AtA12 * AtA3 * AtA10
    inv9 = -AtA0 * AtA9 * AtA15 + \
           AtA0 * AtA11 * AtA13 + \
           AtA8 * AtA1 * AtA15 - \
           AtA8 * AtA3 * AtA13 - \
           AtA12 * AtA1 * AtA11 + \
           AtA12 * AtA3 * AtA9
    inv13 = AtA0 * AtA9 * AtA14 - \
            AtA0 * AtA10 * AtA13 - \
            AtA8 * AtA1 * AtA14 + \
            AtA8 * AtA2 * AtA13 + \
            AtA12 * AtA1 * AtA10 - \
            AtA12 * AtA2 * AtA9
    inv2 = AtA1 * AtA6 * AtA15 - \
           AtA1 * AtA7 * AtA14 - \
           AtA5 * AtA2 * AtA15 + \
           AtA5 * AtA3 * AtA14 + \
           AtA13 * AtA2 * AtA7 - \
           AtA13 * AtA3 * AtA6
    inv6 = -AtA0 * AtA6 * AtA15 + \
           AtA0 * AtA7 * AtA14 + \
           AtA4 * AtA2 * AtA15 - \
           AtA4 * AtA3 * AtA14 - \
           AtA12 * AtA2 * AtA7 + \
           AtA12 * AtA3 * AtA6
    inv10 = AtA0 * AtA5 * AtA15 - \
            AtA0 * AtA7 * AtA13 - \
            AtA4 * AtA1 * AtA15 + \
            AtA4 * AtA3 * AtA13 + \
            AtA12 * AtA1 * AtA7 - \
            AtA12 * AtA3 * AtA5
    inv14 = -AtA0 * AtA5 * AtA14 + \
            AtA0 * AtA6 * AtA13 + \
            AtA4 * AtA1 * AtA14 - \
            AtA4 * AtA2 * AtA13 - \
            AtA12 * AtA1 * AtA6 + \
            AtA12 * AtA2 * AtA5
    inv3 = -AtA1 * AtA6 * AtA11 + \
           AtA1 * AtA7 * AtA10 + \
           AtA5 * AtA2 * AtA11 - \
           AtA5 * AtA3 * AtA10 - \
           AtA9 * AtA2 * AtA7 + \
           AtA9 * AtA3 * AtA6
    inv7 = AtA0 * AtA6 * AtA11 - \
           AtA0 * AtA7 * AtA10 - \
           AtA4 * AtA2 * AtA11 + \
           AtA4 * AtA3 * AtA10 + \
           AtA8 * AtA2 * AtA7 - \
           AtA8 * AtA3 * AtA6
    inv11 = -AtA0 * AtA5 * AtA11 + \
            AtA0 * AtA7 * AtA9 + \
            AtA4 * AtA1 * AtA11 - \
            AtA4 * AtA3 * AtA9 - \
            AtA8 * AtA1 * AtA7 + \
            AtA8 * AtA3 * AtA5
    inv15 = AtA0 * AtA5 * AtA10 - \
            AtA0 * AtA6 * AtA9 - \
            AtA4 * AtA1 * AtA10 + \
            AtA4 * AtA2 * AtA9 + \
            AtA8 * AtA1 * AtA6 - \
            AtA8 * AtA2 * AtA5
    D = AtA0 * inv0 + AtA1 * inv4 \
        + AtA2 * inv8 + AtA3 * inv12
    D = tf.where(tf.abs(D) < EPS, EPS * tf.ones_like(D), D)
    D = tf.expand_dims(D, -1)
    inv0 = tf.expand_dims(inv0, -1)
    inv1 = tf.expand_dims(inv1, -1)
    inv2 = tf.expand_dims(inv2, -1)
    inv3 = tf.expand_dims(inv3, -1)
    inv4 = tf.expand_dims(inv4, -1)
    inv5 = tf.expand_dims(inv5, -1)
    inv6 = tf.expand_dims(inv6, -1)
    inv7 = tf.expand_dims(inv7, -1)
    inv8 = tf.expand_dims(inv8, -1)
    inv9 = tf.expand_dims(inv9, -1)
    inv10 = tf.expand_dims(inv10, -1)
    inv11 = tf.expand_dims(inv11, -1)
    inv12 = tf.expand_dims(inv12, -1)
    inv13 = tf.expand_dims(inv13, -1)
    inv14 = tf.expand_dims(inv14, -1)
    inv15 = tf.expand_dims(inv15, -1)
    inv_AtA = tf.concat([inv0, inv1, inv2, inv3,
                         inv4, inv5, inv6, inv7,
                         inv8, inv9, inv10, inv11,
                         inv12, inv13, inv14, inv15], -1)
    inv_AtA = inv_AtA / D
    return tf.reshape(inv_AtA, [batch_size, npoint, flow_dim // group_size, group_size + 1, group_size + 1])


def get_st_surface(GROUP_SIZE, batch_size, duplicate_grouped_time, flow_dim, new_points_, npoint, nsample, W=None):
    EPS = 0.000001
    new_points_ = tf.concat([new_points_, tf.ones(shape=[batch_size, npoint, flow_dim // GROUP_SIZE, nsample, 1], dtype=new_points_.dtype)], -1)

    if W is None:
        AtA = tf.matmul(new_points_, new_points_, transpose_a=True)
        inv_AtA = get_4D_inversion(AtA, GROUP_SIZE, batch_size, npoint, flow_dim, EPS)
        n_new_points_ = tf.matmul(new_points_, duplicate_grouped_time, transpose_a=True)
        n_new_points_ = tf.matmul(inv_AtA, n_new_points_)
    else:
        AtA = W * new_points_
        AtA = tf.matmul(new_points_, AtA, transpose_a=True)
        inv_AtA = get_4D_inversion(AtA, GROUP_SIZE, batch_size, npoint, flow_dim, EPS)
        n_new_points_ = W * duplicate_grouped_time
        n_new_points_ = tf.matmul(new_points_, n_new_points_, transpose_a=True)
        n_new_points_ = tf.matmul(inv_AtA, n_new_points_)
    return n_new_points_
    
def dynamic_module(xyz, time, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope,
                   module_type='ind_without_time', fps=True, bn=True, pooling='max', knn=False, use_xyz=True,
                   use_nchw=False, delta_t=1, flow_dim=2, last_flow=None, last_W=None, sample_idx=None):
    data_format = 'NCHW' if use_nchw else 'NHWC'
    batch_size = xyz.get_shape()[0].value
    with tf.variable_scope(scope) as sc:
        if sample_idx is None:
            if fps:
                sample_idx = farthest_point_sample(npoint, xyz)
            else:
                sample_idx = tf.tile(tf.expand_dims(tf.range(npoint, dtype=tf.int32), 0), [batch_size, 1])

        radius[delta_t:] = 0

        new_xyz = gather_point(xyz, sample_idx)  # (batch_size, npoint, 3)
        new_time = gather_point(time, sample_idx)  # (batch_size, npoint, 1)
        time_ = tf.reshape(time, [batch_size, 1, -1])  # (batch_size, 1, ndataset)
        new_time_ = tf.abs(time_ - new_time)  # (batch_size, npoint, ndataset)
        radius_ = tf.gather(radius, tf.cast(new_time_, tf.int32))  # (batch_size, npoint, ndataset)
        
        idx, pts_cnt = query_ball_point_var_rad(radius_, nsample, xyz, new_xyz)

        grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
        grouped_time = group_point(time, idx)  # (batch_size, npoint, nsample * 3, channel)
        '''
        FRAME_CNT = 32
        grouped_time -= tf.tile(tf.expand_dims(new_time, 2), [1, 1, nsample, 1])  # time-shift normalization
        grouped_time += FRAME_CNT
        '''
        if points is not None:
            new_points = gather_point(points, sample_idx)  # (batch_size, npoint, channel)
            grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample * 3, channel)
            if use_xyz:
                if module_type == 'ind_without_time':
                    new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
                elif module_type == 'ind':
                    new_points = tf.concat([grouped_xyz, grouped_time, grouped_points],
                                           axis=-1)  # (batch_size, npoint, nample * 3, 3+1+channel)
                else:
                    points_expand = tf.tile(tf.expand_dims(new_points, 2), [1, 1, nsample, 1])
                    new_points = tf.concat([grouped_xyz, grouped_time, grouped_points, points_expand],
                                           axis=-1)  # (batch_size, npoint, nsample * 3, 3+1+channel+channel)
            else:
                new_points = grouped_points
        else:
            new_points = grouped_xyz
        # '''
        flow_list = []
        # Point Feature Embedding
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=False, trainable=False, freeze_bn=True,  # activation_fn=tf.nn.swish,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)

            GROUP_SIZE = 3
            flow_dim = num_out_channel // 2 // GROUP_SIZE * GROUP_SIZE + GROUP_SIZE
            new_points_ = tf_util.conv2d(tf.concat([new_points, grouped_time], -1),
                                         flow_dim - 3, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=False, is_training=is_training, #activation_fn=tf.nn.leaky_relu,
                                         scope='conv%d_dynamic' % i, bn_decay=bn_decay,
                                         data_format=data_format)

            new_points_ = tf.concat([new_points_, grouped_xyz], -1)
            new_points_ = tf.Print(new_points_, [new_points_.shape], "new_points_: ", summarize=2333, first_n=1)

            new_points_ = tf.reshape(new_points_, [batch_size, npoint, nsample, flow_dim // GROUP_SIZE, GROUP_SIZE])
            duplicate_grouped_time = tf.tile(grouped_time, [1, 1, 1, flow_dim // GROUP_SIZE])
            duplicate_grouped_time = tf.reshape(duplicate_grouped_time, [batch_size, npoint, nsample, flow_dim // GROUP_SIZE, 1])
            new_points_ = tf.transpose(new_points_, [0, 1, 3, 2, 4])
            duplicate_grouped_time = tf.transpose(duplicate_grouped_time, [0, 1, 3, 2, 4])
                
            if i == 0 and last_W is None:    
                n_new_points_ = get_st_surface(GROUP_SIZE, batch_size, duplicate_grouped_time, flow_dim, new_points_, npoint, nsample)
            else:
                if i == 0:
                    last_W_shape = last_W.shape
                    W = last_W[:,:,0,:] #tf.reduce_max(last_W, 2)
                    W = tf.reshape(W, [batch_size, last_W_shape[0] // batch_size, -1])
                    W = group_point(W, idx)
                    W = tf.reshape(W, [batch_size * npoint, nsample, last_W_shape[1], last_W_shape[3]])
                    W = tf.transpose(W, [0, 2, 1, 3])
                    
                with tf.variable_scope("pointwise_attention_dynamic", reuse=tf.AUTO_REUSE):
                    W = tf_util.conv2d(W, flow_dim // GROUP_SIZE, [1, 1],
                                   padding='VALID', stride=[1, 1],
                                   bn=False, is_training=is_training, data_format="NCHW",
                                   scope='W_%d_dynamic' % i, bn_decay=bn_decay)
                    W = tf_util.conv2d(W, GROUP_SIZE * 2, [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=False, is_training=is_training, data_format="NHWC", 
                                       scope='W1_%d_dynamic' % i, bn_decay=bn_decay)
                    W = tf_util.conv2d(W, 1, [1, 1],
                                       padding='VALID', stride=[1, 1], data_format="NHWC",
                                       bn=False, is_training=is_training, activation_fn=None,
                                       scope='W2_%d_dynamic' % i, bn_decay=bn_decay)
                W = tf.sigmoid(W)
                W = tf.reshape(W, [batch_size, npoint, flow_dim // GROUP_SIZE, nsample, 1])
                n_new_points_ = get_st_surface(GROUP_SIZE, batch_size, duplicate_grouped_time, flow_dim, new_points_, npoint, nsample, W)
                
            normals_feat = tf.reshape(n_new_points_, [batch_size, npoint, flow_dim // GROUP_SIZE, GROUP_SIZE + 1])
            normals_feat = normals_feat[:, :, :, 0:GROUP_SIZE]
            normals_feat = tf.concat([normals_feat, -tf.ones(shape=[batch_size, npoint, flow_dim // GROUP_SIZE, 1], dtype=normals_feat.dtype)], -1)
            normals_feat = tf.nn.l2_normalize(normals_feat, -1)
            normals_feat = tf.reshape(normals_feat, [batch_size, npoint, -1])
            
            flow_list.append(normals_feat)
            
            new_points_normals = tf.reshape(n_new_points_, [batch_size, npoint, flow_dim // GROUP_SIZE, 1, GROUP_SIZE + 1]) * tf.concat([new_points_, tf.ones(shape=[batch_size, npoint, flow_dim // GROUP_SIZE, nsample, 1], dtype=new_points_.dtype)], -1)
            new_points_normals = tf.reduce_sum(new_points_normals, -1, keep_dims=True) - duplicate_grouped_time
            new_points_normals = tf.reshape(new_points_normals, [batch_size, npoint, flow_dim // GROUP_SIZE, nsample])
            n_new_points_ = tf.reshape(n_new_points_, [batch_size, npoint, flow_dim // GROUP_SIZE, GROUP_SIZE + 1])
            n_new_points_ = n_new_points_[:, :, :, 0:GROUP_SIZE]
            n_new_points_ = tf.concat([n_new_points_, -tf.ones(shape=[batch_size, npoint, flow_dim // GROUP_SIZE, 1], dtype=n_new_points_.dtype)], -1)
            normalization_factor = tf.square(n_new_points_)
            normalization_factor = tf.reduce_sum(normalization_factor, 3, keep_dims=True)
            t = new_points_normals / normalization_factor
            t = tf.reshape(t, [batch_size, npoint, flow_dim // GROUP_SIZE, nsample, 1])
            n_new_points_ = tf.reshape(n_new_points_, [batch_size, npoint, flow_dim // GROUP_SIZE, 1, GROUP_SIZE + 1])
            W = n_new_points_ * t
            # batch_size, npoint, flow_dim // GROUP_SIZE, nsample, GROUP_SIZE
            W = tf.reshape(W, [batch_size * npoint, flow_dim // GROUP_SIZE, nsample, GROUP_SIZE + 1])
            
            
        new_points = tf.reduce_max(new_points, axis=[2], name='maxpool')
        # new_xyz = tf.Print(new_xyz, [new_xyz.shape], "new_xyz: ", summarize=2333, first_n=1)
        # new_time = tf.Print(new_time, [new_time.shape], "new_time: ", summarize=2333, first_n=1)
        # new_points = tf.Print(new_points, [new_points.shape], "new_points: ", summarize=2333, first_n=1)
        # idx = tf.Print(idx, [idx.shape], "idx: ", summarize=2333, first_n=1)

        return new_xyz, new_time, new_points, idx, flow_list, W


def flow_merge_across_res(flow1_list, flow2_list, flow1_idx, flow2_idx, flow1_time, flow2_time, l1_xyz, l2_xyz, scope, is_training, bn_decay, bn=False, data_format='NHWC'):
    it_num = max(len(flow1_list), len(flow2_list))
    for i in range(it_num):
        if i < len(flow1_list):
            flow1_current = flow1_list[i]        
            flow1_channel = flow1_current.shape[-1]
            flow1 = tf_util.conv1d(tf.concat([flow1_current, flow1_time], -1),
                                   flow1_channel, 1, padding='VALID', stride=1,
                                   bn=bn, is_training=is_training, #activation_fn=tf.nn.leaky_relu,
                                   scope='cur1_%d_dynamic' % i, bn_decay=bn_decay,
                                   data_format=data_format)
            flow1 = flow1 + tf_util.conv1d(tf.concat([last_flow1, flow1_time], -1),
                                         flow1_channel, 1,
                                         padding='VALID', stride=1,
                                         bn=bn, is_training=is_training, #activation_fn=tf.nn.leaky_relu,
                                         scope='last_conv1%d_dynamic' % i, bn_decay=bn_decay,
                                         data_format=data_format) if i > 0 else flow1
            last_flow1 = flow1

        if i < len(flow2_list):
            flow2_current = flow2_list[i]        
            flow2_channel = flow2_current.shape[-1]
            flow2 = tf_util.conv1d(tf.concat([flow2_current, flow2_time], -1),
                                   flow2_channel, 1, padding='VALID', stride=1,
                                   bn=bn, is_training=is_training, #activation_fn=tf.nn.leaky_relu,
                                   scope='cur2_%d_dynamic' % i, bn_decay=bn_decay,
                                   data_format=data_format)
            flow2 = flow2 + tf_util.conv1d(tf.concat([last_flow2, flow2_time], -1),
                                         flow2_channel, 1,
                                         padding='VALID', stride=1,
                                         bn=bn, is_training=is_training, #activation_fn=tf.nn.leaky_relu,
                                         scope='last2_%d_dynamic' % i, bn_decay=bn_decay,
                                         data_format=data_format) if i > 0 else flow2
            last_flow2 = flow2

        flow2_to_1 = pointnet_fp_module(l1_xyz, l2_xyz, tf.concat([flow1, flow1_time], -1),
                                        tf.concat([flow2, flow2_time], -1), (flow1_channel, ),
                                        is_training=is_training, bn_decay=bn_decay, scope="2to1_%d_dynamic" % i, bn=bn)
        last_flow1 = last_flow1 + flow2_to_1

        flow1_to_2 = pointnet_fp_module(l2_xyz, l1_xyz, tf.concat([flow2, flow2_time], -1),
                                        tf.concat([flow1, flow1_time], -1), (flow2_channel, ),
                                        is_training=is_training, bn_decay=bn_decay, scope="1to2_%d_dynamic" % i, bn=bn)
        last_flow2 = last_flow2 + flow1_to_2

    return last_flow1, last_flow2

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2, keepdims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1

