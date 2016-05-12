import tensorflow as tf
import numpy as np
import os
import time
import json
import skin
import skimage.io
import scipy.misc

from inception import inception_model


IMAGENET_CKPT = '/home/kuprel/skin/graphs/inception-v3/model.ckpt-157585'
# SKIN_CKPT = '/home/kuprel/skin/nets/brettnet47/checkpoints/SAVE.ckpt-15000'


class ConvNet(object):


    def __init__(self, path, hypes=None, labels=None):
        self.path = path
        self.name = path.split('/')[-1]
        if os.path.exists(self.path):
            self.restore()
        else:
            os.makedirs(self.path)
            os.makedirs(self.path + '/checkpoints')
            self.labels = labels
            self.hypes = hypes
            self.save()


    def save(self):
        """Save net
        """

        with open(self.path + '/hypes.txt', 'w') as f:
            json.dump(self.hypes, f, indent=4)
        with open(self.path + '/labels.txt', 'w') as f:
            json.dump(self.labels, f, indent=4)


    def restore(self):
        """Restore net
        """

        with open(self.path+'/hypes.txt') as f:
            self.hypes = json.load(f)
        with open(self.path+'/labels.txt') as f:
            self.labels = json.load(f)
        self.ckpt = tf.train.get_checkpoint_state(
            self.path+'/checkpoints')

        if self.ckpt:
            ckpt_path = self.ckpt.model_checkpoint_path
            print('found checkpoint: {}'.format(ckpt_path))


    def update_hypes(self):
        """Update hyperparameters, allows for tuning while training"""
        hypes_old = self.hypes
        with open(self.path+'/hypes.txt') as f:
            self.hypes = json.load(f)
        for i in hypes_old:
            if hypes_old[i] != self.hypes[i]:
                print i, 'changed from', hypes_old[i], 'to', self.hypes[i]


    def load_deploy_graph(self, step=1e9, graph_path=None):

        steps = np.sort([
            int(i.split('.')[0].split('step')[-1])
            for i in os.listdir(self.path+'/deploy_graphs')
        ])
        step = steps[np.argmin(np.abs(steps-step))]

        print 'using step', step

        graph_def = tf.GraphDef()
        f = self.path + '/deploy_graphs/step{}.pb'.format(step)
        with open(f) as f: graph_def.ParseFromString(f.read())

        graph = tf.Graph()

        with graph.as_default():
            tf.import_graph_def(graph_def, name='')

        return graph, step


    def predict(self, img, step=1e9):

        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        graph, step = self.load_deploy_graph(step)
        p = graph.get_tensor_by_name('logits:0')
        p = tf.squeeze(tf.nn.softmax(p))

        with tf.Session(graph=graph) as sess:
            P = p.eval({'inputs/batch:0': [img]})

        for i in np.argsort(P)[::-1]:
            if P[i] > 0.05:
                label = self.labels[i].title()
                print '{}% {}'.format(int(P[i]*100), label)


    def get_saliency(self, img, step=1e9):

        graph, step = self.load_deploy_graph(step)
        z = graph.get_tensor_by_name('logits:0')
        x = graph.get_tensor_by_name('input:0')

        s = tf.gradients(tf.reduce_max(z), x)[0]
        s = tf.squeeze(s)
        s = tf.reduce_sum(s*s, reduction_indices=2)
        s = tf.sqrt(s)
        s -= tf.reduce_min(s)
        s /= tf.reduce_max(s)

        with tf.Session(graph=graph) as sess:
            S = s.eval({'inputs/batch:0': [img]})

        scipy.misc.imsave('/tmp/sal.png', 1-S.clip(0, 1))


    def create_deploy_graph(self, output_name='loss/loss', batch_size=1,
        save_path=None):

        filenames, labels = ['f']*10, [0]*10

        graph = tf.Graph()
        with graph.as_default():
            fetches = self.build_graph(filenames, labels, 'val')

            restore_vars = tf.train.ExponentialMovingAverage(0.9)
            restore_vars = restore_vars.variables_to_restore()
            restorer = tf.train.Saver(restore_vars)

        steps = np.sort([
            int(i.split('-')[-1])
            for i in os.listdir(self.path+'/checkpoints')
            if 'save.ckpt' in i and '.meta' not in i
            and 'tempstate' not in i
        ])
        step = steps[-1]

        print 'Creating deploy graph for step', step

        coord = tf.train.Coordinator()

        with tf.Session(graph=graph) as sess:

            tf.initialize_all_variables().run()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            ckpt_path = self.path+'/checkpoints/save.ckpt-'+str(step)
            restorer.restore(sess, ckpt_path)

            v2c = tf.python.client.graph_util.convert_variables_to_constants
            deploy_graph_def = v2c(
                sess, graph.as_graph_def(), [output_name])
            if not os.path.exists(self.path+'/deploy_graphs'):
                os.makedirs(self.path+'/deploy_graphs')
            if save_path is None:
                save_path = self.path + '/deploy_graphs/step{}.pb'.format(step)
            with open(save_path, 'wb') as f:
                f.write(deploy_graph_def.SerializeToString())

        del graph, sess


    def forward(self, filenames, output_name='logits', gpu=None, batch_size=1,
        step=None):

        dev = '/cpu:0'
        if gpu: dev = '/gpu:%d'%gpu

        with tf.device(dev):
            graph, step = self.load_deploy_graph(step=step)

        z = graph.get_tensor_by_name(output_name+':0')
        z = tf.squeeze(z)

        with tf.Session(graph=graph) as sess:

            outputs = np.array([
                z.eval({'inputs/batch:0': [i]})
                for i in filenames
            ])

        return outputs

    def validate(self, data, gpu=0, gpu_fraction=0.8):
        """Validate net

        Args:
            data (list): Dataset as list of tuples, see ``data.load_data``
            gpu_fraction (float): Fraction of GPU memory to use
        """

        config = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        config = tf.ConfigProto(gpu_options=config)

        with tf.device('/gpu:%d'%gpu):
            graph, step = self.load_deploy_graph()

        l = graph.get_tensor_by_name('loss/loss:0')
        z = graph.get_tensor_by_name('logits:0')
        z = tf.squeeze(z)

        with tf.Session(graph=graph, config=config) as sess:

            logits, labels, loss = [], [], []

            for i, (filename, label) in enumerate(data):

                label = self.labels.index(label)

                feed = {
                    'inputs/batch:0': [filename],
                    'inputs/batch:1': [label]
                }

                L, Z = sess.run([l, z], feed)

                loss.append(L)
                logits.append(Z)
                labels.append(label)

                if i % 100 == 0:
                    print ', '.join([
                        'Evaluated %d of %d images' % (i, len(data)),
                        'Train Step: %d' % step,
                        'Loss: %.2f' % np.mean(loss)
                    ])
                    loss = []

        skin.util.make_plots(logits, labels, self.labels,
            os.path.join(self.path, 'plots', 'val', step))

        return logits


    def train(self, data, gpu=0, gpu_fraction=0.8, max_epochs=1e9):
        """Train net

        Args:
            data (list): Dataset as list of tuples [(filename, label string)]
            gpu_fraction (float): Fraction of GPU memory to use
        """

        feed_hypes = [
            'learning_rate', 'epsilon', 'beta1', 'beta2',
            'loc_net_reg', 'xform_reg', 'variable_averages_decay'
        ]

        subset = 'train'
        batch_size = self.hypes['batch_size']

        filenames = [x for x, y in data]
        labels = [self.labels.index(y) for x, y in data]

        logits_matrix, logits_info = skin.util.init_logits(
            self.path, filenames, labels, subset)

        graph = tf.Graph()
        with graph.as_default(), tf.device('/gpu:%d'%gpu):
            fetches = self.build_graph(
                filenames, labels, subset, feed_hypes)

            savers = {
                'short': tf.train.Saver(max_to_keep=3, name='save_short'),
                'long':  tf.train.Saver(max_to_keep=100, name='save_long')
            }
            summary_op = tf.merge_all_summaries()

        coord = tf.train.Coordinator()
        writer = tf.train.SummaryWriter(self.path, flush_secs=5)
        writer.add_graph(tf.get_default_graph())

        config = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        config = tf.ConfigProto(gpu_options=config, allow_soft_placement=True)

        with tf.Session(config=config, graph=graph) as sess:

            feed = {i+':0': self.hypes[i] for i in feed_hypes}
            tf.initialize_all_variables().run(feed)

            if getattr(self, 'ckpt', None):
                ckpt_path = self.ckpt.model_checkpoint_path
                savers['short'].restore(sess, ckpt_path)
            else:
                ckpt_path = IMAGENET_CKPT
                restore_vars = tf.get_collection('_variables_to_restore_')
                if self.hypes['spatial_transformer']:
                    restore_vars = [
                        i for i in restore_vars
                        if 'loc_net' not in i.name
                    ]

                restorer = tf.train.Saver(restore_vars)
                restorer.restore(sess, ckpt_path)

            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print 'number of threads', len(threads)

            try:

                num_epochs = 0

                while not coord.should_stop() and num_epochs < max_epochs:

                    t = time.time()

                    self.update_hypes()
                    feed = {i+':0': self.hypes[i] for i in feed_hypes}

                    fetch_names = sorted(fetches.keys())
                    fetched = sess.run([fetches[i] for i in fetch_names], feed)
                    fetched = dict(zip(fetch_names, fetched))

                    dt = time.time()-t

                    step = fetched['global_step']

                    I = [
                        logits_info['filenames'].index(i)
                        for i in fetched['filenames']
                    ]
                    logits_matrix[I] = fetched['logits']

                    num_epochs = step*batch_size*1.0/len(data)

                    print ', '.join([
                        'Step: %d' % step,
                        'Epochs: %.3f' % num_epochs,
                        'CrossEntropy: %.2f' % fetched['loss'],
                        'RegLoss: %.2f' % fetched['reg_loss'],
                        'FPS: %.1f' % (batch_size/dt)
                    ])

                    if step%5 == 0:

                        print('saving')

                        savers['short'].save(
                            sess = sess,
                            save_path = self.path+'/checkpoints/save.ckpt',
                            global_step = fetches['global_step']
                        )

                        writer.add_summary(summary_op.eval(feed), step)

                        skin.util.save_matrix(self.path, logits_matrix,
                            'logits', subset, step)
                        skin.util.make_plots(
                            logits_matrix, logits_info['labels'], self.labels,
                            os.path.join(self.path, 'plots', subset, step))

                    if step%10 == 0:

                        self.create_deploy_graph()

                        savers['long'].save(
                            sess = sess,
                            save_path = self.path+'/checkpoints/SAVE.ckpt',
                            global_step = fetches['global_step']
                        )

            except Exception as e: coord.request_stop(e)

        coord.join(threads)


    def build_graph(self, filenames, labels, subset, feed_hypes=None):

        hypes = self.hypes.copy()

        if feed_hypes:
            with tf.name_scope(None):
                for i in feed_hypes:
                    hypes[i] = tf.placeholder('float32', name=i)
                    hypes[i].set_shape([])

        with tf.name_scope('inputs'):

            filenames, labels = tf.train.slice_input_producer(
                tensor_list = [filenames, labels],
                capacity = hypes['batch_size']*2,
                shuffle = (subset=='train')
            )

            filenames, labels = tf.train.batch(
                tensor_list = [filenames, labels],
                capacity = hypes['batch_size']*2,
                batch_size = hypes['batch_size']
            )

            images0 = [
                tf.image.decode_jpeg(tf.read_file(i[0]), channels=3)
                for i in tf.split(0, hypes['batch_size'], filenames)
            ]

            images0 = [skin.util.square_pad(i) for i in images0]

            if subset == 'train':
                images0 = [tf.image.random_flip_left_right(i) for i in images0]
                images0 = [tf.image.random_flip_up_down(i) for i in images0]

            if hypes['spatial_transformer']:
                images = skin.util.spatial_tranform(
                    images0, hypes['batch_size'], subset,
                    hypes['loc_net'], hypes['xform_reg'])
            else:
                images = tf.pack([
                    tf.image.resize_images(i, 299, 299)
                    for i in images0
                ])

            with tf.name_scope(None):
                images = tf.identity(images, name='input')

        logits, logits_aux = inception_model.inference(
            images = (images-128)/128.,
            num_classes = len(self.labels),
            for_training = (subset == 'train'),
            restore_logits = (subset != 'train')
        )

        with tf.name_scope(None):
            logits = tf.identity(logits, name='logits')
        tf.histogram_summary('logits', logits)

        with tf.name_scope('loss'):

            batch_size, num_classes = logits.get_shape().as_list()

            labels_sparse = tf.sparse_to_dense(
                sparse_indices = tf.transpose(
                    tf.pack([tf.range(batch_size), labels])
                ),
                output_shape = [batch_size, num_classes],
                sparse_values = np.ones(batch_size, dtype='float32')
            )

            loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels_sparse)
            loss = tf.reduce_mean(loss, name='loss')

            loss_aux = tf.nn.softmax_cross_entropy_with_logits(
                logits_aux, labels_sparse)
            loss_aux = tf.reduce_mean(loss_aux, name='loss_aux')

            loss = 0.7*loss + 0.3*loss_aux

            tf.scalar_summary('loss', loss)

        fetches = {
            'loss': loss,
            'filenames': filenames,
            'logits': logits
        }

        def print_graph_ops():
            with open('/tmp/graph_ops.txt', 'w') as f:
                for op in tf.get_default_graph().get_operations():
                    f.write(op.type.ljust(35)+'\t'+op.name+'\n')

        if subset == 'train':

            reg_losses = tf.get_collection('regularization_losses')

            for i, j in enumerate(reg_losses):
                if 'loc_net' in j.name:
                    reg_losses[i] *= hypes['loc_net_reg']

            reg_loss = tf.add_n(reg_losses)
            tf.scalar_summary('reg_loss', reg_loss)

            with tf.variable_scope('reg_loss'):
                loss += reg_loss

            print_graph_ops()

            global_step = tf.Variable(0, name='global_step', trainable=False)

            opt = eval('tf.train.{}Optimizer'.format('Adam'))(
                learning_rate = hypes['learning_rate'],
                epsilon = hypes['epsilon'],
                beta1 = hypes['beta1'],
                beta2 = hypes['beta2']
            )

            grads = opt.compute_gradients(loss)
            apply_grads = opt.apply_gradients(grads, global_step)

            variable_averages = tf.train.ExponentialMovingAverage(
                hypes['variable_averages_decay'], global_step)
            variables_to_average = (tf.trainable_variables() +
                                    tf.moving_average_variables())
            variables_averages_op = variable_averages.apply(variables_to_average)

            batchnorm_updates_op = tf.group(*tf.get_collection('_update_ops_'))

            train_op = tf.group(apply_grads, variables_averages_op, batchnorm_updates_op)

            for grad, var in grads:
                tf.histogram_summary(var.op.name, var)
                try:
                    tf.histogram_summary(var.op.name + '/gradients', grad)
                except:
                    print var.op.name

            fetches.update({
                'reg_loss': reg_loss,
                'train_op': train_op,
                'global_step': global_step
            })

        else:

            print_graph_ops()

        return fetches
