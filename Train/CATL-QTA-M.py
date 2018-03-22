import mxnet as mx
import numpy as np
import codecs, json
import os, h5py, sys, argparse
import time
import argparse
import bisect
import random
import gensim
import logging


class maxone(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        index_max = np.argmax(x,axis = 1)
        y = np.zeros(x.shape)
        for j in range(x.shape[0]):
            y[j,index_max[j]] = 1
        self.assign(out_data[0], req[0], mx.nd.array(y))
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = out_grad[0].asnumpy()
        self.assign(in_grad[0], req[0], mx.nd.array(y))
@mx.operator.register("maxone")
class maxoneProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(maxoneProp, self).__init__(need_top_grad=False)
    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], [] 
    def create_operator(self, ctx, shapes, dtypes):
        return maxone()
    
    
class VQAtrainIter(mx.io.DataIter):
    def __init__(self, img0, img1, nmt, sentences,  answer, qtype, qtype_ans, batch_size, buckets=[10,20,30], invalid_label=-1,
                 text_name='text-kk', nmt_name = 'nmt',
                  img1_name = 'image1',label0_name='softmax_label',label1_name='softmax-qtype_label', dtype='float32',
                 layout='TNC'):
        super(VQAtrainIter, self).__init__()
        if not buckets:
            buckets = [i for i, j in enumerate(np.bincount([len(s) for s in sentences]))
                       if j >= batch_size]
        buckets.sort()

        ndiscard = 0
        self.data = [[] for _ in buckets]
        self.nmt = [[] for _ in buckets]
        self.img0 = [[] for _ in buckets]
        self.img1 = [[] for _ in buckets]
        self.label0 = [[] for _ in buckets]
        self.label1 = [[] for _ in buckets]
        self.qtype = [[] for _ in buckets]
        for i in range(len(sentences)):
            
            buck = bisect.bisect_left(buckets, len(sentences[i]))
            
            if buck == len(buckets):
                ndiscard += 1
                continue
            buff = np.full((buckets[buck],), invalid_label, dtype=dtype)
            buff[:len(sentences[i])] = sentences[i]
            self.data[buck].append(buff)
            self.nmt[buck].append(nmt[i])
            self.img0[buck].append(img0[i])
            self.img1[buck].append(img1[i])
            self.label0[buck].append(answer[i])
            self.label1[buck].append(qtype_ans[i])
            self.qtype[buck].append(qtype[i])
            
        self.data = [np.asarray(i, dtype=dtype) for i in self.data]
        self.nmt = [np.asarray(i, dtype=dtype) for i in self.nmt]
        self.img0 = [np.asarray(i, dtype=dtype) for i in self.img0]
        self.img1 = [np.asarray(i, dtype=dtype) for i in self.img1]
        self.label0 = [np.asarray(i, dtype=dtype) for i in self.label0]
        self.label1 = [np.asarray(i, dtype=dtype) for i in self.label1]
        self.qtype = [np.asarray(i, dtype=dtype) for i in self.qtype]
        
        
        
        print("WARNING: discarded %d sentences longer than the largest bucket."%ndiscard)

        self.batch_size = batch_size
        self.buckets = buckets
        self.text_name = text_name
        self.nmt_name = nmt_name
        self.img1_name = img1_name
        self.label0_name = label0_name
        self.label1_name = label1_name
        self.dtype = dtype
        self.invalid_label = invalid_label
        self.nd_text = []
        self.nd_nmt = []
        self.nd_img0 = []
        self.nd_img1 = []
        self.ndlabel = []
        self.nd_qtype = []
        self.major_axis = layout.find('N')
        self.default_bucket_key = max(buckets)

        if self.major_axis == 0:
            self.provide_data = [mx.io.DataDesc(name='nmt', shape=(batch_size,300), layout='NTC'),
                mx.io.DataDesc(name='text-kk', shape=(batch_size, 30), layout='NTC'),
                mx.io.DataDesc('image1',(batch_size,2048),layout='NTC'),
                mx.io.DataDesc('image0',(batch_size,2048,),layout='NTC'),
                    mx.io.DataDesc('qtype',(batch_size,12),layout='NTC')]
            self.provide_label = [(label_name, (batch_size, ))]
        elif self.major_axis == 1:
            self.provide_data = [mx.io.DataDesc(name='nmt', shape=(300,batch_size), layout='TNC'),
                mx.io.DataDesc(name='text-kk', shape=( 30,batch_size), layout='TNC'),
                mx.io.DataDesc('image1',(2048,batch_size),layout='TNC'),
                mx.io.DataDesc('image0',(2048,batch_size),layout='TNC'),
                mx.io.DataDesc('qtype',(12,batch_size),layout='TNC')]
            self.provide_label = [(label0_name, (batch_size,)),
                                  (label1_name, (batch_size,))]
        else:
            raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")

        self.idx = []
        for i, buck in enumerate(self.data):
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0

        self.reset()

    def reset(self):
        self.curr_idx = 0
        self.nd_text = []
        self.nd_nmt = []
        self.nd_img0 = []
        self.nd_img1 = []
        self.ndlabel0 = []
        self.ndlabel1 = []
        self.nd_qtype = []
        for i,buck in enumerate(self.data):
            
            self.nd_text.append(mx.ndarray.array(buck, dtype=self.dtype)) 
            self.nd_nmt.append(mx.ndarray.array(self.nmt[i], dtype=self.dtype))
            self.nd_img0.append(mx.ndarray.array(self.img0[i], dtype=self.dtype))
            self.nd_img1.append(mx.ndarray.array(self.img1[i], dtype=self.dtype))
            self.ndlabel0.append(mx.ndarray.array(self.label0[i], dtype=self.dtype))
            self.ndlabel1.append(mx.ndarray.array(self.label1[i], dtype=self.dtype))
            self.nd_qtype.append(mx.ndarray.array(self.qtype[i], dtype=self.dtype))
            

    def next(self):
        if self.curr_idx == len(self.idx):
            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        if self.major_axis == 1:
            nmt = self.nd_nmt[i][j:j + self.batch_size].T
            img0 = self.nd_img0[i][j:j + self.batch_size].T
            img1 = self.nd_img1[i][j:j + self.batch_size].T
            text = self.nd_text[i][j:j + self.batch_size].T
            qtype = self.nd_qtype[i][j:j+self.batch_size].T
            label0 = self.ndlabel0[i][j:j+self.batch_size]
            label1 = self.ndlabel1[i][j:j+self.batch_size]
        else:
            nmt = self.nd_nmt[i][j:j + self.batch_size]
            img0 = self.nd_img0[i][j:j + self.batch_size]
            img1 = self.nd_img1[i][j:j + self.batch_size]
            text = self.nd_text[i][j:j + self.batch_size]
            label0 = self.ndlabel0[i][j:j+self.batch_size]
            label1 = self.ndlabel1[i][j:j+self.batch_size]
            qtype = self.nd_qtype[i][j:j+self.batch_size]
        
        data = [  nmt, text, img0, img1]
        label1 = mx.nd.Reshape(label1,(self.batch_size,))
        return mx.io.DataBatch(data, [label0,label1],
                         bucket_key=self.buckets[i],
                           provide_data=[
                mx.io.DataDesc(name='nmt', shape=nmt.shape,layout='TNC'),
                mx.io.DataDesc(name='text-kk', shape=text.shape, layout='TNC'),
                mx.io.DataDesc(name='image0', shape=img0.shape, layout='TNC'),
                mx.io.DataDesc(name='image1', shape=img1.shape, layout='TNC'),
                ],
                         provide_label=[(self.label0_name, label0.shape),
                                       (self.label1_name, label1.shape)])


def load_data_with_word2vec(question,dataset,model):
    word2vec_question = np.zeros((len(question),300))
    for q in range(len(question)):
        sample = question[q,:]
        vec = []
        count = 0
        for i in sample:
            if i != 0:
                
                word = dataset[str(int(i)+1)]
                if word in model:
                    count = count + 1
                    if len(vec) == 0:
                        vec.append(np.array(model[word]))
                    else:
                        vec = vec + np.array(model[word])
        word2vec_question[q,:] = vec[0]/count
    return word2vec_question



numgpus = 8
parser = argparse.ArgumentParser(description="VQA",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--out-dim1', type=int, default=8000,
                    help='max num of epochs')
parser.add_argument('--out-dim2', type=int, default=4000,
                    help='max num of epochs')

parser.add_argument('--num-hidden1', type=int, default=2000,
                    help='max num of epochs')
parser.add_argument('--num-hidden2', type=int, default=2000,
                    help='max num of epochs')

parser.add_argument('--dropout', type=float, default=0.3,
                    help='max num of epochs')
parser.add_argument('--lr', type=float, default= 0.001,
                    help='max num of epochs')

parser.add_argument('--batch-size', type=int, default=256*numgpus,
                    help='the batch size.')

def eval_metrics(): 
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [mx.metric.Accuracy(),mx.metric.CrossEntropy()]:
        eval_metrics.add(child_metric)
    return eval_metrics

def evaluation_callback(iter_no, sym, arg, aux):
    
    infile = r'/home/ubuntu/efs/multi-task/multitask1-hidden%f_drop%f_lr%f_sgd.log' %(args.num_hidden2,args.dropout,args.lr)
    with open(infile) as f:
        f = f.readlines()
        val_acc = []
        val_acc_key = ["Validation-multi-accuracy-task0"]
        for line in f:
            for phrase in val_acc_key:
                if phrase in line:
                    idx = line.index('=')
                    val_acc.append(line[idx+1:len(line)-1])
        if len(val_acc) == 1:
            max_val_acc = float(val_acc[0])
        else:
            max_val_acc = max([float(item) for item in val_acc[:-1]])
                    
        if float(val_acc[-1]) > max_val_acc:
            max_val_acc = val_acc[-1]
            mx.model.save_checkpoint('/home/ubuntu/efs/multi-task/multitask1-hidden%f_drop%f_lr%f_sgd' %(args.num_hidden2,args.dropout,args.lr), iter_no, sym, arg, aux)
            
class Multi_Accuracy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self, num=None):
        self.num = num
        super(Multi_Accuracy, self).__init__('multi-accuracy')

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0 if self.num is None else [0] * self.num
        self.sum_metric = 0.0 if self.num is None else [0.0] * self.num

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        if self.num is not None:
            assert len(labels) == self.num

        for i in range(len(labels)):
            pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            if self.num is None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
            else:
                self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                self.num_inst[i] += len(pred_label.flat)

    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num is None:
            return super(Multi_Accuracy, self).get()
        else:
            return zip(*(('%s-task%d'%(self.name, i), float('nan') if self.num_inst[i] == 0
                                                      else self.sum_metric[i] / self.num_inst[i])
                       for i in range(self.num)))

    def get_name_value(self):
        """Returns zipped name and value pairs.
        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        if self.num is None:
            return super(Multi_Accuracy, self).get_name_value()
        name, value = self.get()
        return list(zip(name, value))


def train(args):
    print(args.dropout)
    logging.basicConfig(filename='/home/ubuntu/efs/multi-task/multitask1-hidden%f_drop%f_lr%f_sgd.log' %(args.num_hidden2,args.dropout, args.lr), level=logging.INFO)
    logging.info('Start!')
    
    print 'loading lstm questions...'
    input_ques_h5 = '/home/ubuntu/efs/top1480ans/data_prepro.h5'
    train_data = {}
    val_data = {}
    def right_align(seq,lengths):
        v = np.zeros(np.shape(seq))
        N = np.shape(seq)[1]
        for i in range(np.shape(seq)[0]):
            v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
        return v


    with h5py.File(input_ques_h5,'r') as hf:
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        tem = hf.get('ques_val')
        val_data['question'] = np.array(tem)-1

        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        tem = hf.get('ques_length_val')
        val_data['length_q'] = np.array(tem)


        train_data['question'] = right_align(train_data['question'], train_data['length_q'])
        val_data['question'] = right_align(val_data['question'], val_data['length_q'])
        print(max(train_data['length_q']))
        print(max(val_data['length_q']))

    train_question = train_data['question']
    val_question = val_data['question']
    
    
    
    
    model = gensim.models.KeyedVectors.load_word2vec_format('/home/ubuntu/efs/GoogleNews-vectors-negative300.bin.gz', binary=True)
    with open('/home/ubuntu/efs/top1480ans/itoword.json') as json_data:
        d = json.load(json_data)
        train_q_w2v = load_data_with_word2vec(train_question,d['ix_to_word'],model)
        val_q_w2v = load_data_with_word2vec(val_question,d['ix_to_word'],model)

    print 'loading rcnn feature...'
    train_img_f = np.load('/home/ubuntu/efs/top1480ans/rcnn_2048_image_train_feature.npz')
    val_img_f = np.load('/home/ubuntu/efs/top1480ans/rcnn_2048_image_val_feature.npz')
    train_img0 = train_img_f['x']
    val_img0 = val_img_f['x']
    
    print 'loading resnet feature...'
    train_img_f = np.load('/home/ubuntu/efs/top1480ans/resnet_avgpool_train_feature.npz')
    val_img_f = np.load('/home/ubuntu/efs/top1480ans/resnet_avgpool_val_feature.npz')
    train_img1 = train_img_f['x']
    val_img1 = val_img_f['x']

    print 'loading vqa answers...'
    train_ans = np.loadtxt("/home/ubuntu/efs/top1480ans/train_answer_num.txt")
    val_ans = np.loadtxt("/home/ubuntu/efs/top1480ans/val_answer_num.txt")
    
    print("loading question type...")
    train_qtype = np.load("/home/ubuntu/efs/top1480ans/train_question_type.npz")['arr_0']
    val_qtype = np.load("/home/ubuntu/efs/top1480ans/val_question_type.npz")['arr_0']

    print("loading question type answer ...")
    train_qtype_ans = np.load('/home/ubuntu/efs/predict_qtype/qtype_train_number.npy')
    val_qtype_ans = np.load('/home/ubuntu/efs/predict_qtype/qtype_val_number.npy')
    
    
    seq_len = 26
    num_filter = [512,1]
    workspace = 1024
    ################################################
    layout = 'TN'
    
    vocabulary_size = 7498
    data_train  = VQAtrainIter(train_img0, train_img1, train_q_w2v, train_question, train_ans, train_qtype, train_qtype_ans, args.batch_size,layout=layout)

    
    np.random.seed(2)
    idx = np.random.choice(range(538543), 1000, replace=False)
    val_img0 = val_img0[idx]
    val_img1 = val_img1[idx]
    val_q_w2v = val_q_w2v[idx]
    val_question = val_question[idx]
    val_ans = val_ans[idx]
    val_qtype = val_qtype[idx]
    val_qtype_ans = val_qtype_ans[idx]

    data_eva = VQAtrainIter(val_img0, val_img1, val_q_w2v, val_question, val_ans, val_qtype, val_qtype_ans,args.batch_size, layout=layout)
    
    print("loading params")
    sym, arg_params, aux_params = mx.model.load_checkpoint('concat-resnet-rcnn-w2v-lstm_hidden8192_drop0.300000_lr0.010000',918)
    new_args = dict({k:arg_params[k] for k in arg_params})  
    

    cell = mx.rnn.SequentialRNNCell()
    for i in range(2):
        cell.add(mx.rnn.FusedRNNCell(1024, num_layers=1,
                                         mode='lstm', prefix='lstm_l%d'%i,
                                         bidirectional=False))
    
    def sym_gen(seq_len):
        text = mx.sym.Variable('text-kk')
        nmt_data = mx.sym.Variable('nmt')
        label = mx.sym.Variable('softmax_label')
        image1_data = mx.sym.Variable('image1')
        image0_data = mx.sym.Variable('image0')
        
    
        nmt_data = mx.sym.transpose(nmt_data)
        image0_data = mx.sym.transpose(image0_data)
        image1_data = mx.sym.transpose(image1_data)
    
        
        embed = mx.sym.Embedding(data=text, input_dim=vocabulary_size,output_dim=100,name='embed')
        output, _ = cell.unroll(seq_len, inputs=embed, merge_outputs=True, layout='TNC')
        LSTM = mx.sym.SequenceLast(data = output)
        
        fc3 = mx.sym.FullyConnected(data = LSTM, num_hidden=args.num_hidden2, no_bias = True,name='fc-new2')
        act3 = mx.symbol.Activation(data = fc3, name='relu-new2', act_type="relu")
        bn3 = mx.sym.BatchNorm(data = act3, name = 'bn-new2')
        drop3 = mx.sym.Dropout(bn3,p = args.dropout)
        net_final = mx.symbol.FullyConnected(data=drop3, num_hidden=12, name='fc-qtype')
        net_final = mx.symbol.SoftmaxOutput(data=net_final, name='softmax-qtype')

        net_final_argmax = mx.sym.Custom(net_final,name='maxone', op_type='maxone')
        mask = mx.sym.FullyConnected(data = net_final_argmax, num_hidden=2048*2)
        mask = mx.sym.softmax(mask,axis=1)

        concat1 = mx.sym.Concat(image0_data,image1_data,dim = 1)
        concat1 = concat1*mask
        concat1 = mx.sym.L2Normalization(data = concat1)
        
        concat1 = mx.sym.Concat(concat1,nmt_data,dim = 1)

        
        concat2 = mx.sym.Concat(concat1,LSTM,dim = 1)
        concat2 = mx.sym.Reshape(concat2, shape =(args.batch_size/numgpus, 300+4096+1024))

        fc1 = mx.sym.FullyConnected(data = concat2, num_hidden=8192, no_bias = True, name = 'fc1')
        act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
        bn1 = mx.sym.BatchNorm(data = act1, name = 'bn4')
        drop = mx.sym.Dropout(bn1,p = args.dropout)
        fc2 = mx.sym.FullyConnected(data = drop, num_hidden=1480, name = 'fc2')
        net = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

        net_group = mx.sym.Group([net, net_final])
        return net_group, ('nmt','text-kk','image1','image0'),('softmax_label','softmax-qtype_label')
    
    
    
    mod = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = 30,
        context             = [mx.gpu(i) for i in range(numgpus)])
    
    
    data_shapes = [     
           mx.io.DataDesc(name='nmt', shape=(300,args.batch_size ), layout='TNC'),    
        mx.io.DataDesc(name='text-kk', shape=(30,args.batch_size ), layout='TNC'),
                mx.io.DataDesc('image0',(2048,args.batch_size),layout='TNC'),
    mx.io.DataDesc('image1',(2048,args.batch_size),layout='TNC'),
    ]

    label_shapes = [
                   mx.io.DataDesc(
                    'softmax_label',
                    (args.batch_size,),
                    layout='N'),
                   mx.io.DataDesc(
                    'softmax-qtype_label',
                    (args.batch_size,),
                    layout='N'),
                   ]
    mod.bind(data_shapes=data_shapes, label_shapes = label_shapes, for_training=True, force_rebind = True, grad_req="write")

    print('Start training...')
    epoch_size = 1115299/args.batch_size
    
    lr_sch = mx.lr_scheduler.MultiFactorScheduler(step = [epoch_size * x  for x in [10,20,40,50,60,75,90,100,125,150,175,200,225,250,275,300,320,340,360,380,400,450,500,550,600,700,750,800,900,1000]], factor=0.7)
    mod.fit(data_train, data_eva, num_epoch=5000, 
            eval_metric= Multi_Accuracy(num = 2),
            arg_params=new_args,
            aux_params=aux_params,
            allow_missing=True,   
            optimizer='sgd', 
            optimizer_params=(('learning_rate', args.lr), ('lr_scheduler', lr_sch), ('momentum', 0.9),),
            epoch_end_callback=evaluation_callback
           )    
    
    
if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    
    train(args)
