import tensorflow as tf
import numpy as np
import os
import time
import json
import skin
from tensorflow.python.client import graph_util
from matplotlib import pyplot
from skimage import io
from scipy import misc


class ConvNet(object):


    def __init__(self, name, hypes=None, labels=None):
        self.name = name
        if os.path.exists(self.name):
            self._restore()
            if hypes:
                self.hypes = hypes
                self._save()
        else:
            os.makedirs(self.name)
            os.makedirs(self.name + '/checkpoints')
            self.labels = labels
            self.hypes = hypes
            self._save()


    def _save(self):
        """Save net
        """

        with open(self.name + '/hypes.txt', 'w') as f:
            json.dump(self.hypes, f, indent=4)
        with open(self.name + '/labels.txt', 'w') as f:
            json.dump(self.labels, f, indent=4)


    def _restore(self):
        """Restore net
        """

        with open(self.name+'/hypes.txt') as f: self.hypes = json.load(f)
        with open(self.name+'/labels.txt') as f: self.labels = json.load(f)
        self.ckpt = tf.train.get_checkpoint_state(self.name+'/checkpoints')

        if self.ckpt:
            ckpt_path = self.ckpt.model_checkpoint_path
            print('found checkpoint: {}'.format(ckpt_path))


    def predict(self, url=None, filename=None, step=None):

        if url is not None:
            img = io.imread(url)
        if filename is not None:
            img = io.imread(filename)
        if step is None:
            step = lambda i: int(i.split('.')[0].split('step')[-1])
            steps = os.listdir(self.name+'/deploy_graphs')
            steps = sorted([step(i) for i in steps])
            step = steps[-1]
        
        print 'using step', step

        graph_def = tf.GraphDef()
        f = self.name + '/deploy_graphs/step{}.pb'.format(step)
        with open(f) as f: graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        g = tf.get_default_graph()

        if self.hypes['base_net'] == 'inception_v1':
            x = g.get_tensor_by_name('inputs/batch:0')
            logits = g.get_tensor_by_name('logits_2/logits:0')
            img = misc.imresize(img, [224]*2+[3]).astype('float32')
            img -= 117
        if self.hypes['base_net'] == 'inception_v3':
            x = g.get_tensor_by_name('inputs/batch:0')
            logits = g.get_tensor_by_name('logits/logits:0')
            img = misc.imresize(img, [299]*2+[3]).astype('float32')
            img -= 128
            img /= 128

        logits = tf.squeeze(logits)
        

        with tf.Session() as sess:
            preds = logits.eval({x:[img]})

        preds = np.exp(preds)
        preds /= preds.sum()

        I = np.argsort(preds)[::-1]

        for i in np.argsort(preds)[::-1]:
            if preds[i] > 0.05:
                p = int(preds[i]*100)
                label = self.labels[i].title()
                print '{}% {}'.format(p, label)


    def get_saliency(self, url=None, filename=None, step=None, 
        output_file='/tmp/saliency.svg', gpu=2):

        if url is not None:
            img0 = io.imread(url)
        if filename is not None:
            img0 = io.imread(filename)
        if step is None:
            step = lambda i: int(i.split('.')[0].split('step')[-1])
            steps = os.listdir(self.name+'/deploy_graphs')
            steps = sorted([step(i) for i in steps])
            step = steps[-1]
            print 'using step', step

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

        graph_def = tf.GraphDef()
        f = self.name + '/deploy_graphs/step{}.pb'.format(step)
        with open(f) as f: graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        g = tf.get_default_graph()

        if self.hypes['base_net'] == 'inception_v1':
            x = g.get_tensor_by_name('inputs/batch:0')
            logits = g.get_tensor_by_name('logits_2/logits:0')
            img0 = misc.imresize(img0, [224]*2+[3])
            img = img0.astype('float32')
            img -= 117
        if self.hypes['base_net'] == 'inception_v3':
            x = g.get_tensor_by_name('inputs/batch:0')
            logits = g.get_tensor_by_name('logits/logits:0')
            img0 = misc.imresize(img0, [299]*2+[3])
            img = img0.astype('float32')
            img -= 128
            img /= 128

        logits = tf.squeeze(logits)

        s = tf.gradients(tf.reduce_max(logits), x)[0]
        s = tf.squeeze(s)
        s = tf.reduce_sum(s*s, reduction_indices=2)
        s = tf.sqrt(s)

        with tf.Session() as sess:
            S = s.eval({x:[img]})

        S -= S.min()
        S /= S.max()
        S = 255*(1-S)
        S = S.clip(0, 255).astype('uint8')

        fig, ax = pyplot.subplots(1,2)
        ax[0].imshow(img0, interpolation='nearest')
        ax[1].imshow(S, cmap='gray', interpolation='nearest')
        fig.savefig(output_file)


    def train(self, data, gpu, gpu_fraction):
        """Train net
        
        Args:
            data (list): Dataset as list of tuples, see ``data.load_data``
            gpu_fraction (float): Fraction of GPU memory to use
        """

        phase = 'train'
        batch_size = self.hypes['batch_size']
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

        filenames = [x for x, y in data]
        labels = [self.labels.index(y) for x, y in data]

        logits_matrix, logits_info = self._init_logits(filenames, labels, phase)

        graph = tf.Graph()
        with graph.as_default():
            fetches = skin.inference.build_graph(self.hypes, filenames, 
                labels, len(self.labels), phase, batch_size)
            savers = {
                'short': tf.train.Saver(max_to_keep=3, name='save_short'),
                'long':  tf.train.Saver(max_to_keep=100, name='save_long')
            }
            summary_op = tf.merge_all_summaries()

        features_dim = fetches['features'].get_shape().as_list()[-1]
        features_matrix = np.zeros(
            shape = [len(logits_matrix), features_dim],
            dtype = 'float16')
        features_matrix[:] = np.nan

        coord = tf.train.Coordinator()
        writer = tf.train.SummaryWriter(self.name)
        writer.add_graph(tf.get_default_graph().as_graph_def())

        print_str = ', '.join([
            'Step: %d',
            'Loss: %.2f',
            'Weights norm: %.2f',
            'Time/image (ms): %.1f'
        ])

        config = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        config = tf.ConfigProto(gpu_options=config)

        with tf.Session(config=config, graph=graph) as sess:

            tf.initialize_all_variables().run()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print 'number of threads', len(threads)

            if getattr(self, 'ckpt', None):
                ckpt_path = self.ckpt.model_checkpoint_path
                savers['short'].restore(sess, ckpt_path)

            try:

                while not coord.should_stop():

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
                    features_matrix[I] = fetched['features']

                    print print_str % (
                        step,
                        fetched['loss'],
                        fetched['weights_norm'],
                        dt/batch_size*1000
                    )

                    if step%100 == 0:

                        print('saving')

                        savers['short'].save(
                            sess = sess, 
                            save_path = self.name+'/checkpoints/save.ckpt',
                            global_step = fetches['global_step']
                        )

                        writer.add_summary(sess.run(summary_op), 
                            global_step=step)

                        self._save_logits(logits_matrix, phase, step)
                        self.make_plots(logits_matrix, logits_info, phase, step)

                    if step%1000 == 0:

                        savers['long'].save(
                            sess = sess, 
                            save_path = self.name+'/checkpoints/SAVE.ckpt',
                            global_step = fetches['global_step']
                        )
                        self._save_features(features_matrix, phase, step)

            except Exception as e: coord.request_stop(e)

        coord.join(threads)


    def test(self, data, gpu, gpu_fraction):
        """Test net

        Args:
            data (list): Dataset as list of tuples, see ``data.load_data``
            gpu_fraction (float): Fraction of GPU memory to use
        """

        phase = 'test'
        batch_size = 6
        steps_evaluated = []
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
            
        filenames = [x for x, y in data]
        labels = [self.labels.index(y) for x, y in data]

        logits_matrix, logits_info = self._init_logits(filenames, labels, phase)
        
        graph = tf.Graph()
        with graph.as_default():
            fetches = skin.inference.build_graph(self.hypes, filenames, 
                labels, len(self.labels), phase, batch_size)
            saver = tf.train.Saver()

        features_dim = fetches['features'].get_shape().as_list()[-1]
        features_matrix = np.zeros(
            shape = [len(logits_matrix), features_dim],
            dtype = 'float16')
        features_matrix[:] = np.nan

        coord = tf.train.Coordinator()

        print_str = ', '.join([
            'Evaluated %d of %d images',
            'Train Step: %d',
            'Loss: %.2f',
            'Time/image (ms): %.1f'
        ])

        num_examples = len(data)

        config = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        config = tf.ConfigProto(gpu_options=config)

        with tf.Session(config=config, graph=graph) as sess:

            tf.initialize_all_variables().run()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            while True:

                step = lambda i: int(i.split('-')[-1])
                steps = sorted([
                    step(i) for i in os.listdir(self.name+'/checkpoints')
                    if 'save.ckpt' in i and '.meta' not in i
                    and 'tempstate' not in i
                    and step(i) not in steps_evaluated
                ])

                if len(steps):
                    step = steps[-1]
                    print 'Evaluating checkpoint from step {}'.format(step)
                    steps_evaluated.append(step)
                else:
                    print 'No checkpoints to evaluate, trying again in 1 minute'
                    time.sleep(60)
                    continue

                saver.restore(sess, self.name+'/checkpoints/save.ckpt-'+str(step))
                if self.hypes['base_net'] == 'inception_v1':
                    output_name = 'logits_2/logits'
                if self.hypes['base_net'] == 'inception_v3':
                    output_name = 'logits/logits'
                deploy_graph_def = graph_util.convert_variables_to_constants(sess, 
                    graph.as_graph_def(), [output_name])
                if not os.path.exists(self.name+'/deploy_graphs'):
                    os.makedirs(self.name+'/deploy_graphs')
                f = self.name + '/deploy_graphs/step{}.pb'.format(step)
                with open(f, 'wb') as f:
                    f.write(deploy_graph_def.SerializeToString())

                logits_matrix[:] = np.nan
                features_matrix[:] = np.nan
                num_evaluated = 0

                try:

                    while (
                        num_evaluated < num_examples + batch_size
                        and not coord.should_stop()
                    ):

                        t = time.time()

                        fetch_names = sorted(fetches.keys())
                        fetched = sess.run([fetches[i] for i in fetch_names])
                        fetched = dict(zip(fetch_names, fetched))

                        dt = time.time()-t

                        I = [
                            logits_info['filenames'].index(i) 
                            for i in fetched['filenames']
                        ]
                        logits_matrix[I] = fetched['logits']
                        features_matrix[I] = fetched['features']

                        print print_str % (
                            num_evaluated,
                            num_examples,
                            step,
                            fetched['loss'],
                            dt/batch_size*1000
                        )

                        num_evaluated += batch_size

                    print sum(np.isnan(logits_matrix.sum(1))), 'missed images'
                    self._save_logits(logits_matrix, phase, step)
                    self.make_plots(logits_matrix, logits_info, phase, step)
                    if step%1000 == 0: 
                        self._save_features(features_matrix, phase, step)

                except Exception as e:
                    coord.request_stop(e)

        coord.request_stop()
        coord.join(threads)


    def make_plots(self, logits, logits_info, phase, step, confidence_interval=0.8):
        """Make analysis plots and save to disk

        Args:
            phase (string): Either `train` or `test`
            step (int): Training iteration to analyze
            confidence_interval (float): Confidence interval [0,1] of
                uncertainty ellipses. If `None`, no ellipses are plotted
        """

        def _save_fig(fig, category, fig_type):
            fig_path = os.path.join(self.name, 'plots', phase, category, fig_type)
            if not os.path.exists(fig_path): os.makedirs(fig_path)
            fig.savefig(fig_path + '/step{}.svg'.format(step))

        with open('../data/competition.txt') as f:
            competition_data = json.load(f)

        b = np.logical_not(np.isnan(logits.sum(1)))
        Y, Z = np.array(logits_info['labels'])[b], logits[b].astype('float32')

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

            
    def _init_logits(self, filenames, labels, phase):

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

        with open(self.name + '/logits_info_{}.txt'.format(phase), 'w') as f:
            json.dump(logits_info, f, indent=4)

        return logits_matrix, logits_info


    def _save_logits(self, logits, phase, step):

        path = os.path.join(self.name, 'logits', phase)
        if not os.path.exists(path): os.makedirs(path)
        np.save(path+'/step{}.npy'.format(step), logits)


    def _save_features(self, features, phase, step):

        path = os.path.join(self.name, 'features', phase)
        if not os.path.exists(path): os.makedirs(path)
        np.save(path+'/step{}.npy'.format(step), features)