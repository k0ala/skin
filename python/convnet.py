import tensorflow as tf
import numpy as np
import os
import time
import json
import skin
import skimage.io
import scipy.misc
from matplotlib import pyplot

from inception import inception_model


IMAGENET_CKPT = '/home/kuprel/skin/graphs/inception-v3/model.ckpt-157585'
SKIN_CKPT = '/home/kuprel/skin/nets/brettnet42/checkpoints/SAVE.ckpt-7000'


class ConvNet(object):


    def __init__(self, path, hypes=None, labels=None):
        self.path = path
        self.name = path.split('/')[-1]
        if os.path.exists(self.path):
            self.restore()
            if hypes:
                self.hypes = hypes
                self.save()
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

        with open(self.path+'/hypes.txt') as f: self.hypes = json.load(f)
        with open(self.path+'/labels.txt') as f: self.labels = json.load(f)
        self.ckpt = tf.train.get_checkpoint_state(self.path+'/checkpoints')

        if self.ckpt:
            ckpt_path = self.ckpt.model_checkpoint_path
            print('found checkpoint: {}'.format(ckpt_path))


    def load_deploy_graph(self, step=1e9):

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
            fetches = skin.inference.build_graph(self.hypes, filenames, 
                labels, len(self.labels), subset='val', batch_size=batch_size)

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

        var2const = tf.python.client.graph_util.convert_variables_to_constants
        coord = tf.train.Coordinator()

        with tf.Session(graph=graph) as sess:

            tf.initialize_all_variables().run()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            ckpt_path = self.path+'/checkpoints/save.ckpt-'+str(step)
            restorer.restore(sess, ckpt_path)
            deploy_graph_def = var2const(
                sess, graph.as_graph_def(), [output_name])
            if not os.path.exists(self.path+'/deploy_graphs'):
                os.makedirs(self.path+'/deploy_graphs')
            if save_path is None:
                save_path = self.path + '/deploy_graphs/step{}.pb'.format(step)
            with open(save_path, 'wb') as f:
                f.write(deploy_graph_def.SerializeToString())

        del graph, sess


    def validate(self, data, gpu, gpu_fraction):
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
                        'Evaluated %d of %d images',
                        'Train Step: %d',
                        'Loss: %.2f'
                    ]) % (i, len(data), step, np.mean(loss))
                    loss = []

            logits = np.array(logits, dtype='float16')
            labels = np.array(labels, dtype='int32')

            self._save_matrix(logits, 'logits', 'validation', step)
            self.make_plots(logits, labels, 'validation', step)


    def train(self, data, gpu, gpu_fraction, max_epochs=1e9):
        """Train net
        
        Args:
            data (list): Dataset as list of tuples [(filename, label string)]
            gpu_fraction (float): Fraction of GPU memory to use
        """

        subset = 'train'
        batch_size = self.hypes['batch_size']

        filenames = [x for x, y in data]
        labels = [self.labels.index(y) for x, y in data]

        logits_matrix, logits_info = self._init_logits(filenames, labels, subset)

        graph = tf.Graph()
        with graph.as_default(), tf.device('/gpu:%d'%gpu):
            fetches = skin.inference.build_graph(self.hypes, filenames, 
                labels, len(self.labels), subset, batch_size)

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

            tf.initialize_all_variables().run()

            if getattr(self, 'ckpt', None):
                ckpt_path = self.ckpt.model_checkpoint_path
                savers['short'].restore(sess, ckpt_path)
            else:
                if self.hypes['spatial_transformer']:
                    ckpt_path = SKIN_CKPT
                    restore_vars = [
                        i for i in tf.all_variables()
                        if 'spatial_transform' not in i.name
                    ]
                    # restore_vars = {}
                    # for i in tf.all_variables():
                    #     name = i.op.name
                    #     if 'spatial_transform' in name:
                    #         name = '/'.join(name.split('/')[1:])
                    #     restore_vars[name] = i
                else:
                    ckpt_path = IMAGENET_CKPT
                    restore_vars = tf.get_collection('_variables_to_restore_')

                restorer = tf.train.Saver(restore_vars)
                restorer.restore(sess, ckpt_path)

            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print 'number of threads', len(threads)

            try:

                num_epochs = 0

                while not coord.should_stop() and num_epochs < max_epochs:

                    t = time.time()

                    fetch_names = sorted(fetches.keys())
                    fetched = sess.run([fetches[i] for i in fetch_names])
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
                        'WeightNorm: %.2f' % fetched['reg_loss'],
                        'FPS: %.1f' % (batch_size/dt)
                    ])

                    if step%100 == 0:

                        print('saving')

                        savers['short'].save(
                            sess = sess, 
                            save_path = self.path+'/checkpoints/save.ckpt',
                            global_step = fetches['global_step']
                        )

                        writer.add_summary(summary_op.eval(), step)

                        self._save_matrix(logits_matrix, 'logits', subset, step)
                        self.make_plots(logits_matrix, logits_info['labels'],
                            subset, step)

                    if step%1000 == 0:

                        self.create_deploy_graph()

                        savers['long'].save(
                            sess = sess, 
                            save_path = self.path+'/checkpoints/SAVE.ckpt',
                            global_step = fetches['global_step']
                        )

            except Exception as e: coord.request_stop(e)

        coord.join(threads)


    def make_plots(self, logits, labels, subset, step, confidence_interval=0.8):
        """Make analysis plots and save to disk

        Args:
            subset (string): Either `train` or `val`
            step (int): Training iteration to analyze
            confidence_interval (float): Confidence interval [0,1] of
                uncertainty ellipses. If `None`, no ellipses are plotted
        """

        def _save_fig(fig, category, fig_type):
            fig_path = os.path.join(self.path, 'plots', subset, category, fig_type)
            if not os.path.exists(fig_path): os.makedirs(fig_path)
            fig.savefig(fig_path + '/step{}.svg'.format(step))

        with open('../data/competition.txt') as f:
            competition_data = json.load(f)

        b = np.logical_not(np.isnan(logits.sum(1)))
        Y, Z = np.array(labels)[b], logits[b].astype('float32')

        for category in ['pigmented', 'epidermal']:

            i = self.labels.index('{} benign'.format(category))
            j = self.labels.index('{} malignant'.format(category))

            if category not in competition_data: competition_data[category] = None

            IJ = np.logical_or(Y==i, Y==j)
            y = Y[IJ].copy()
            y[y==i] = 0
            y[y==j] = 1

            p = np.exp(Z[IJ][:,[i,j]])
            p = p[:,1]/p.sum(axis=1)

            hist_fig = skin.analysis.get_hist(p, y)
            _save_fig(hist_fig, category, 'hist')

            roc_fig = skin.analysis.get_roc(p, y, competition_data[category], confidence_interval)
            _save_fig(roc_fig, category, 'roc')

            pyplot.close('all')

            
    def _init_logits(self, filenames, labels, subset):

        logits_info = {i: j for i, j in zip(filenames, labels)}
        logits_info = {
            'filenames': logits_info.keys(), 
            'labels': logits_info.values()
        }

        logits_matrix = np.zeros(
            shape = [len(logits_info['filenames']), len(self.labels)],
            dtype = 'float16'
        )
        logits_matrix[:] = np.nan

        with open(self.path + '/logits_info_{}.txt'.format(subset), 'w') as f:
            json.dump(logits_info, f, indent=4)

        return logits_matrix, logits_info


    def _save_matrix(self, A, name, subset, step):

        path = os.path.join(self.path, name, subset)
        if not os.path.exists(path): os.makedirs(path)
        np.save(path+'/step{}.npy'.format(step), A)
