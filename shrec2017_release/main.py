import os
import sys

from gesture_utils import get_parser, import_class, Recorder, Stat, RandomState
import torch

import yaml
import numpy as np
import tensorflow as tf
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))

sparser = get_parser()
FLAGS = sparser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FRAME = FLAGS.num_frames
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
COMMAND_FILE = FLAGS.command_file

MODALITY = FLAGS.modality
MODEL_PATH = FLAGS.model_path

MODEL = importlib.import_module(FLAGS.network_file) # import network module
MODEL_FILE = os.path.join(FLAGS.network_file+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp *_dataloader*.py %s' % (LOG_DIR)) # bkp of data loader
os.system('cp main.py %s' % (LOG_DIR)) # bkp of train procedure
os.system('cp net_utils.py %s' % (LOG_DIR)) # bkp of net_utils
os.system('cp %s %s' % (COMMAND_FILE, LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
INPUT_DIM = 7

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = RandomState(seed=self.arg.random_seed)
        self.recoder = Recorder(self.arg.work_dir, self.arg.print_log)
        #self.device = GpuDataParallel()
        self.data_loader = {}
        self.topk = (1, 5)
        self.stat = Stat(self.arg.model_args['num_classes'], self.topk)
        #self.model, self.optimizer = self.Loading()
        self.load_data()
        self.loss = self.criterion()

    def criterion(self):
        loss = torch.nn.CrossEntropyLoss(reduction="none")
        return loss #self.device.criterion_to_device(loss)

    def load_data(self):
        print("Loading data")
        Feeder = import_class(self.arg.dataloader)
        self.data_loader = dict()
        if self.arg.train_loader_args != {}:
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_loader_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.arg.num_worker,
            )
        if self.arg.valid_loader_args != {}:
            self.data_loader['valid'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.valid_loader_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.arg.num_worker,
            )
        if self.arg.test_loader_args != {}:
            test_dataset = Feeder(**self.arg.test_loader_args)
            self.stat.test_size = len(test_dataset)
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.arg.num_worker,
            )
        print("Loading data finished.")

    def start(self):
        if self.arg.phase == 'train' and MODALITY == "static":
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            with tf.Graph().as_default():
                #'''
                with tf.device('/gpu:' + str(GPU_INDEX)):
                    pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME, INPUT_DIM)
                    is_training_pl = tf.placeholder(tf.bool, shape=())

                    # Note the global_step=batch parameter to minimize.
                    # That tells the optimizer to helpfully increment the 'batch' parameter
                    # for you every time it trains.
                    batch = tf.get_variable('batch', [],
                                            initializer=tf.constant_initializer(0), trainable=False)
                    bn_decay = get_bn_decay(batch)
                    tf.summary.scalar('bn_decay', bn_decay)

                    # Get model and loss
                    pred, end_points = MODEL.get_model(pointclouds_pl, NUM_FRAME, is_training_pl,
                                                       bn_decay=bn_decay, CLS_COUNT=self.arg.model_args['num_classes'])
                    MODEL.get_loss(pred, labels_pl, end_points)
                    losses = tf.get_collection('losses')
                    total_loss = tf.add_n(losses, name='total_loss')
                    tf.summary.scalar('total_loss', total_loss)
                    for l in losses:
                        tf.summary.scalar(l.op.name, l)
                    correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
                    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
                    tf.summary.scalar('accuracy', accuracy)

                    print("--- Get training operator")
                    # Get training operator
                    learning_rate = get_learning_rate(batch)
                    tf.summary.scalar('learning_rate', learning_rate)
                    if OPTIMIZER == 'momentum':
                        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                    elif OPTIMIZER == 'adam':
                        optimizer = tf.train.AdamOptimizer(learning_rate)
                    train_op = optimizer.minimize(total_loss, global_step=batch)

                    # Add ops to save and restore all the variables.
                    self.saver = tf.train.Saver()
                #'''
                # Create a session
                tf_config = tf.ConfigProto()
                tf_config.gpu_options.allow_growth = False
                tf_config.allow_soft_placement = True
                tf_config.log_device_placement = False
                sess = tf.Session(config=tf_config)

                # Add summary writers
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
                test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
                # Init variables
                init = tf.global_variables_initializer()
                sess.run(init)

                pretrained = MODEL_PATH
                if pretrained is not '':
                    variables = tf.contrib.framework.get_variables_to_restore()
                    variables_to_restore = [v for v in variables if '_dynamic' not in v.name and "batch:" not in v.name]
                    if OPTIMIZER == 'adam':
                        variables_to_restore = [v for v in variables_to_restore if "_power" not in v.name]
                    elif OPTIMIZER == 'momentum':
                        variables_to_restore = [v for v in variables_to_restore if "Momentum" not in v.name]
                    print("variables_to_restore", variables_to_restore)
                    loading_saver = tf.train.Saver(variables_to_restore)
                    pretrained_model_path = pretrained #"pretrained_on_modelnet/model1464.ckpt"
                    loading_saver.restore(sess, pretrained_model_path)
                    print("The model has been loaded !!!!!!!!!!!!!")
          

                ops = {'pointclouds_pl': pointclouds_pl,
                       'labels_pl': labels_pl,
                       'is_training_pl': is_training_pl,
                       'pred': pred,
                       'loss': total_loss,
                       'accuracy': accuracy,
                       'train_op': train_op,
                       'merged': merged,
                       'step': batch,
                       'end_points': end_points}
                for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                    save_model = ((epoch + 1) % self.arg.save_interval == 0) or \
                                 (epoch + 1 == self.arg.num_epoch)
                    eval_model = ((epoch + 1) % self.arg.eval_interval == 0) or \
                                 (epoch + 1 == self.arg.num_epoch)
                    self.train(epoch, sess, ops, train_writer)
                    if eval_model:
                        if self.arg.valid_loader_args != {}:
                            self.stat.reset_statistic()
                            self.eval(loader_name=['valid'], sess=sess, ops=ops)
                        if self.arg.test_loader_args != {}:
                            self.stat.reset_statistic()
                            self.eval(loader_name=['test'], sess=sess, ops=ops)
                    if save_model:
                        save_path = self.saver.save(sess, os.path.join(LOG_DIR, "model%03d.ckpt" % epoch))
                        log_string("Model saved in file: %s" % save_path)

        if self.arg.phase == 'train' and MODALITY == "dynamic":
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            with tf.Graph().as_default():
                #'''
                with tf.device('/gpu:' + str(GPU_INDEX)):
                    pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME, INPUT_DIM)
                    is_training_pl = tf.placeholder(tf.bool, shape=())

                    # Note the global_step=batch parameter to minimize.
                    # That tells the optimizer to helpfully increment the 'batch' parameter
                    # for you every time it trains.
                    batch = tf.get_variable('batch', [],
                                            initializer=tf.constant_initializer(0), trainable=False)
                    bn_decay = get_bn_decay(batch)
                    tf.summary.scalar('bn_decay', bn_decay)

                    # Get model and loss
                    static_pred, flow_pred, end_points = MODEL.get_model(pointclouds_pl, NUM_FRAME, is_training_pl,
                                                       bn_decay=bn_decay, CLS_COUNT=self.arg.model_args['num_classes'])
                    pred = flow_pred
                    MODEL.get_loss(pred, labels_pl, end_points)
                    losses = tf.get_collection('losses')
                    total_loss = tf.add_n(losses, name='total_loss')
                    tf.summary.scalar('total_loss', total_loss)
                    for l in losses:
                        tf.summary.scalar(l.op.name, l)
                    correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
                    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
                    tf.summary.scalar('accuracy', accuracy)

                    print("--- Get training operator")
                    # Get training operator
                    learning_rate = get_learning_rate(batch)
                    tf.summary.scalar('learning_rate', learning_rate)
                    if OPTIMIZER == 'momentum':
                        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                    elif OPTIMIZER == 'adam':
                        optimizer = tf.train.AdamOptimizer(learning_rate)
                    train_op = optimizer.minimize(total_loss, global_step=batch)

                    # Add ops to save and restore all the variables.
                    self.saver = tf.train.Saver()

                #'''
                # Create a session
                tf_config = tf.ConfigProto()
                tf_config.gpu_options.allow_growth = False
                tf_config.allow_soft_placement = True
                tf_config.log_device_placement = False
                sess = tf.Session(config=tf_config)

                # Add summary writers
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
                test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
                # Init variables
                init = tf.global_variables_initializer()
                sess.run(init)

                if MODEL_PATH is not '':
                    variables = tf.contrib.framework.get_variables_to_restore()
                    variables_to_restore = [v for v in variables if '_dynamic' not in v.name and "batch:" not in v.name]
                    if OPTIMIZER == 'adam':
                        variables_to_restore = [v for v in variables_to_restore if "_power" not in v.name]
                    elif OPTIMIZER == 'momentum':
                        variables_to_restore = [v for v in variables_to_restore if "Momentum" not in v.name]
                    print("variables_to_restore", variables_to_restore)
                    loading_saver = tf.train.Saver(variables_to_restore)
                    pretrained_model_path = MODEL_PATH
                    loading_saver.restore(sess, pretrained_model_path)
                    print("The model has been loaded !!!!!!!!!!!!!")
                    
                ops = {'pointclouds_pl': pointclouds_pl,
                       'labels_pl': labels_pl,
                       'is_training_pl': is_training_pl,
                       'pred': tf.tuple([static_pred, flow_pred]),
                       'loss': total_loss,
                       'accuracy': accuracy,
                       'train_op': train_op,
                       'merged': merged,
                       'step': batch,
                       'end_points': end_points}
                for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                    save_model = ((epoch + 1) % self.arg.save_interval == 0) or \
                                 (epoch + 1 == self.arg.num_epoch)
                    eval_model = ((epoch + 1) % self.arg.eval_interval == 0) or \
                                 (epoch + 1 == self.arg.num_epoch)
                    self.train(epoch, sess, ops, train_writer)
                    if eval_model:
                        if self.arg.valid_loader_args != {}:
                            self.stat.reset_statistic()
                            self.eval_when_training(loader_name=['valid'], sess=sess, ops=ops)
                        if self.arg.test_loader_args != {}:
                            self.stat.reset_statistic()
                            self.eval_when_training(loader_name=['test'], sess=sess, ops=ops)
                    if save_model:
                        save_path = self.saver.save(sess, os.path.join(LOG_DIR, "model%04d.ckpt" % epoch))
                        log_string("Model saved in file: %s" % save_path)

        elif self.arg.phase == 'test':
            with tf.Graph().as_default():
                with tf.device('/gpu:' + str(GPU_INDEX)):
                    pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME, INPUT_DIM)
                    is_training_pl = tf.placeholder(tf.bool, shape=())

                    # Note the global_step=batch parameter to minimize.
                    # That tells the optimizer to helpfully increment the 'batch' parameter
                    # for you every time it trains.
                    batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
                    bn_decay = get_bn_decay(batch)
                    static_pred, flow_pred, end_points = MODEL.get_model(pointclouds_pl, NUM_FRAME, is_training_pl, bn_decay=bn_decay, CLS_COUNT=self.arg.model_args['num_classes'])

                # Create a session
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True
                config.log_device_placement = False
                sess = tf.Session(config=config)
                # Init variables
                loading_saver = tf.train.Saver()
                loading_saver.restore(sess, MODEL_PATH)
                print("The model has been loaded !!!!!!!!!!!!!")
                # Add summary writers
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'sess'), sess.graph)

                ops = {'pointclouds_pl': pointclouds_pl,
                       'labels_pl': labels_pl,
                       'is_training_pl': is_training_pl,
                       'static_pred': static_pred,
                       'flow_pred': flow_pred,
                       'merged': merged,
                       'step': batch}
               
                self.recoder.print_log('Model: {}.'.format(MODEL_PATH))
                result_list = []
                if self.arg.valid_loader_args != {}:
                    static_ratio = 1.0
                    self.stat.reset_statistic()
                    self.mixed_eval(['valid'], sess, ops, writer, static_ratio=static_ratio)
                    result_list.append((static_ratio, self.print_inf_log(self.arg.optimizer_args['start_epoch'], "Valid (static_ratio: %.2f)" % static_ratio)))

                    static_ratio = 0.0
                    self.stat.reset_statistic()
                    self.mixed_eval(['valid'], sess, ops, writer, static_ratio=static_ratio)
                    result_list.append((static_ratio, self.print_inf_log(self.arg.optimizer_args['start_epoch'],
                                                                     "Valid (static_ratio: %.2f)" % static_ratio)))

                    static_ratio = 0.5
                    self.stat.reset_statistic()
                    self.mixed_eval(['valid'], sess, ops, writer, static_ratio=static_ratio)
                    result_list.append((static_ratio, self.print_inf_log(self.arg.optimizer_args['start_epoch'],
                                                                     "Valid (static_ratio: %.2f)" % static_ratio)))

                if self.arg.test_loader_args != {}:
                    static_ratio = 1.0
                    self.stat.reset_statistic()
                    self.mixed_eval(['test'], sess, ops, writer, static_ratio=static_ratio)
                    result_list.append((static_ratio, self.print_inf_log(self.arg.optimizer_args['start_epoch'], "Test (static_ratio: %.2f)" % static_ratio)))

                    static_ratio = 0.0
                    self.stat.reset_statistic()
                    self.mixed_eval(['test'], sess, ops, writer, static_ratio=static_ratio)
                    result_list.append((static_ratio, self.print_inf_log(self.arg.optimizer_args['start_epoch'],
                                                                 "Test (static_ratio: %.2f)" % static_ratio)))

                    static_ratio = 0.5
                    self.stat.reset_statistic()
                    self.mixed_eval(['test'], sess, ops, writer, static_ratio=static_ratio)
                    result_list.append((static_ratio, self.print_inf_log(self.arg.optimizer_args['start_epoch'],
                                                                 "Test (static_ratio: %.2f)" % static_ratio)))

            self.recoder.print_log('Evaluation Done.\n')
            for r in result_list:
                self.recoder.print_log("static ratio %.2f: acc=%.6f" % r)

    def print_inf_log(self, epoch, mode):
        stati = self.stat.show_accuracy('{}/{}_confusion_mat'.format(self.arg.work_dir, mode))
        prec1 = stati[str(self.topk[0])] / self.stat.test_size * 100
        prec5 = stati[str(self.topk[1])] / self.stat.test_size * 100
        self.recoder.print_log("Epoch {}, {}, Evaluation: prec1 {:.4f}, prec5 {:.4f}".
                               format(epoch, mode, prec1, prec5),
                               '{}/{}.txt'.format(self.arg.work_dir, self.arg.phase))
        return prec1

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)


    def train(self, epoch, sess, ops, tf_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = True
        self.recoder.print_log('Training epoch: {}'.format(epoch))
        loader = self.data_loader['train']
        loss_value = []
        self.recoder.timer_reset()
        #current_learning_rate = [group['lr'] for group in self.optimizer.optimizer.param_groups]
        for batch_idx, data in enumerate(loader):
            self.recoder.record_timer("dataloader")
            batch_data = data[0].detach().numpy()
            batch_label = data[1].detach().numpy()
            self.recoder.record_timer("device")
            feed_dict = {ops['pointclouds_pl']: batch_data.reshape(batch_data.shape[0],-1,batch_data.shape[-1]),
                         ops['labels_pl']: batch_label,
                         ops['is_training_pl']: is_training}
            summary, step, _, loss_val, pred_val, acc_val = sess.run([ops['merged'], ops['step'],
                                                                      ops['train_op'], ops['loss'], ops['pred'],
                                                                      ops['accuracy']], feed_dict=feed_dict)
            tf_writer.add_summary(summary, step)
            self.recoder.record_timer("forward")
            loss = torch.from_numpy(np.array(loss_val))
            loss_value.append(loss.item())
            if batch_idx % self.arg.log_interval == 0:
                # self.viz.append_loss(epoch * len(loader) + batch_idx, loss.item())
                self.recoder.print_log(
                    '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}'
                        .format(epoch, batch_idx, len(loader), loss.item()))
                self.recoder.print_time_statistics()
        self.recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
        
    def mixed_eval(self, loader_name, sess, ops, writer, static_ratio):
        is_training = False
        for l_name in loader_name:
            loader = self.data_loader[l_name]
            for batch_idx, data in enumerate(loader):
                cur_batch_data = data[0] #self.device.data_to_device(data[0])
                cur_batch_label = data[1] #self.device.data_to_device(data[1])

                feed_dict = {ops['pointclouds_pl']: cur_batch_data.detach().numpy().reshape(cur_batch_data.shape[0],-1,cur_batch_data.shape[-1]),
                             ops['labels_pl']: cur_batch_label.detach().numpy(),
                             ops['is_training_pl']: is_training}
                pred_val_static, pred_val_flow = sess.run([ops['static_pred'], ops['flow_pred']], feed_dict=feed_dict)
                #summary, step, pred_val_static, pred_val_flow = sess.run([ops['merged'], ops['step'], ops['static_pred'], ops['flow_pred']], feed_dict=feed_dict)
                '''
                pred_prob_static = self.softmax(pred_val_static)
                pred_prob_flow = self.softmax(pred_val_flow)
                '''
                pred_prob_static = pred_val_static
                pred_prob_flow = pred_val_flow
                
                pred_val = pred_prob_static * static_ratio + pred_prob_flow * (1.0 - static_ratio)
                pred_val = self.softmax(pred_val)
                
                #writer.add_summary(summary, step)
                output = torch.from_numpy(pred_val)
                self.stat.update_accuracy(output.data, cur_batch_label, topk=self.topk)
            
    def softmax(self, arr_list):
        ret_list = []
        for arr in arr_list:
            arr = arr - np.amax(arr)
            exp_arr = np.exp(arr)
            ret_list.append(exp_arr / np.sum(exp_arr))
        return np.array(ret_list)
    
    def eval(self, loader_name, sess, ops):
        is_training = False
        for l_name in loader_name:
            loader = self.data_loader[l_name]
            loss_mean = []
            for batch_idx, data in enumerate(loader):
                cur_batch_data = data[0] #self.device.data_to_device(data[0])
                cur_batch_label = data[1] #self.device.data_to_device(data[1])

                feed_dict = {ops['pointclouds_pl']: cur_batch_data.detach().numpy().reshape(cur_batch_data.shape[0],-1,cur_batch_data.shape[-1]),
                             ops['labels_pl']: cur_batch_label.detach().numpy(),
                             ops['is_training_pl']: is_training}
                summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                              ops['loss'], ops['pred']], feed_dict=feed_dict)

                output = torch.from_numpy(pred_val)
                loss_mean += np.array(loss_val).flatten().tolist()
                self.stat.update_accuracy(output.data, cur_batch_label, topk=self.topk)
            self.recoder.print_log('mean loss: ' + str(np.mean(loss_mean)))

    def eval_when_training(self, loader_name, sess, ops):
        is_training = False
        for l_name in loader_name:
            loader = self.data_loader[l_name]
            loss_mean = []
            for batch_idx, data in enumerate(loader):
                cur_batch_data = data[0] #self.device.data_to_device(data[0])
                cur_batch_label = data[1] #self.device.data_to_device(data[1])

                feed_dict = {ops['pointclouds_pl']: cur_batch_data.detach().numpy().reshape(cur_batch_data.shape[0],-1,cur_batch_data.shape[-1]),
                             ops['labels_pl']: cur_batch_label.detach().numpy(),
                             ops['is_training_pl']: is_training}
                summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                              ops['loss'], ops['pred']], feed_dict=feed_dict)
                weight = 0.5
                output_static = torch.from_numpy(pred_val[0])
                output_flow = torch.from_numpy(pred_val[1])
                output = weight * output_static + (1.0 - weight) * output_flow
                
                loss_mean += np.array(loss_val).flatten().tolist()
                self.stat.update_accuracy(output.data, cur_batch_label, topk=self.topk)
            self.recoder.print_log('mean loss: ' + str(np.mean(loss_mean)))


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    if FLAGS.config is not None:
        with open(FLAGS.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(FLAGS).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    processor = Processor(args)
    processor.start()
    LOG_FOUT.close()
