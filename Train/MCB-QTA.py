#from __future__ import print_function
import mxnet as mx
import numpy as np
import codecs, json
import os, h5py, sys, argparse
import time
import logging
import argparse
import sys
import bisect
import random
import gensim

class VQAtrainIterDic(mx.io.DataIter):
    def __init__(self, dic0, dic1, img1, nmt, sentences, answer,  question_type, batch_size, buckets=[10,20,30], invalid_label=-1,
                 text_name='text', nmt_name = 'nmt', img1_name = 'image1', label_name='softmax_label', dtype='float32',
                 layout='NTC'):
        super(VQAtrainIterDic, self).__init__()
        if not buckets:
            buckets = [i for i, j in enumerate(np.bincount([len(s) for s in sentences]))
                       if j >= batch_size]
        buckets.sort()
        ndiscard = 0
        
        self.data = [[] for _ in buckets]
        self.nmt = [[] for _ in buckets]
        self.img1 = [[] for _ in buckets]
        self.label = [[] for _ in buckets]
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
            self.img1[buck].append(img1[i])
            self.label[buck].append(answer[i])
            self.qtype[buck].append(question_type[i])

        self.data = [np.asarray(i, dtype=dtype) for i in self.data]
        self.nmt = [np.asarray(i, dtype=dtype) for i in self.nmt]
        self.img1 = [np.asarray(i, dtype='int32') for i in self.img1]
        self.label = [np.asarray(i, dtype=dtype) for i in self.label]
        self.qtype = [np.asarray(i, dtype=dtype) for i in self.qtype]
        
        print("WARNING: discarded %d sentences longer than the largest bucket."%ndiscard)

        self.batch_size = batch_size
        self.buckets = buckets
        self.text_name = text_name
        self.img1_name = img1_name
        self.nmt_name = nmt_name
        self.label_name = label_name
        self.dtype = dtype
        self.invalid_label = invalid_label
        self.nd_text = []
        self.nd_img1 = []
        self.ndlabel = []
        self.major_axis = layout.find('N')
        self.default_bucket_key = max(buckets)
        self.dic0 = dic0
        self.dic1 = dic1
        if self.major_axis == 0:
            self.provide_data = [mx.io.DataDesc(name='w2v', shape=(batch_size,300), layout='NTC'),
                mx.io.DataDesc(name='text', shape=( batch_size,30), layout='NTC'),
                mx.io.DataDesc('image0',(batch_size,2048,14,14),layout='NTC'),
                mx.io.DataDesc('image1',(batch_size,36,2048),layout='NTC'),
                mx.io.DataDesc('qtype',(batch_size,12),layout='NTC')]
            self.provide_label = [(label_name, (batch_size, self.default_bucket_key))]
        elif self.major_axis == 1:
            self.provide_data = [mx.io.DataDesc(name='w2v', shape=(300,batch_size), layout='TNC'),
                mx.io.DataDesc(name='text', shape=( 30,batch_size), layout='TNC'),
                mx.io.DataDesc('image0',(2048,14,14,batch_size),layout='TNC'),
                mx.io.DataDesc('image1',(36,2048,batch_size),layout='TNC'),
                mx.io.DataDesc('qtype',(12,batch_size),layout='TNC')]
            self.provide_label = [(label_name, (batch_size,))]
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
        self.nd_img1 = []
        self.nd_nmt = []
        self.ndlabel = []
        self.nd_qtype = []
        
        for i,buck in enumerate(self.data):
            
            self.nd_text.append(mx.ndarray.array(buck, dtype=self.dtype)) 
            self.nd_nmt.append(mx.ndarray.array(self.nmt[i], dtype=self.dtype))
            self.nd_img1.append(mx.ndarray.array(self.img1[i], dtype='int32'))
            self.ndlabel.append(mx.ndarray.array(self.label[i], dtype=self.dtype))
            self.nd_qtype.append(mx.ndarray.array(self.qtype[i], dtype=self.dtype))
            

    def next(self):
        if self.curr_idx == len(self.idx):
            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1

        if self.major_axis == 1:
            img0 = mx.nd.zeros((2048,14,14,self.batch_size)) 
            img0_id_list = self.nd_img1[i][j:j+self.batch_size]
            k = 0
            for m in img0_id_list:
                t_ivec = self.dic0.item().get(m)
                img0[:,:,k] = np.reshape(t_ivec,(36,2048,1))
                k = k+1

            img1 = mx.nd.zeros((36,2048,self.batch_size)) 
            img1_id_list = self.nd_img1[i][j:j+self.batch_size]
            k = 0
            for m in img1_id_list:
                t_ivec = self.dic1.item().get(m)
                img1[:,:,k] = np.reshape(t_ivec,(36,2048,1))
                k = k+1
            text = self.nd_text[i][j:j + self.batch_size].T
            nmt = self.nd_nmt[i][j:j+self.batch_size].T
            label = self.ndlabel[i][j:j+self.batch_size]
            qtype = self.nd_qtype[i][j:j + self.batch_size].T
            
        else:
            img0 = mx.nd.zeros((self.batch_size,2048,14,14))
            img0_id_list = self.nd_img1[i][j:j+self.batch_size]
            k = 0
            for m in img0_id_list:
                t_ivec = self.dic0.item().get(int(m.asnumpy()))
                img0[k] = t_ivec
                k = k+1

            img1 = mx.nd.zeros((self.batch_size,36,2048))
            img1_id_list = self.nd_img1[i][j:j+self.batch_size]
            k = 0
            for m in img1_id_list:
                t_ivec = self.dic1.item().get(int(m.asnumpy()))
                img1[k] = t_ivec
                k = k+1
            text = self.nd_text[i][j:j + self.batch_size]
            
            label = self.ndlabel[i][j:j+self.batch_size]
            nmt = self.nd_nmt[i][j:j+self.batch_size]
            qtype = self.nd_qtype[i][j:j + self.batch_size]
            
        
        data = [nmt, text,img0, img1, qtype]
        return mx.io.DataBatch(data, [label],
                         bucket_key=self.buckets[i],
                         provide_data=[
                mx.io.DataDesc(name='w2v', shape=nmt.shape,layout='NTC'),
                mx.io.DataDesc(name='text', shape=text.shape, layout='NTC'),
                mx.io.DataDesc(name='image0', shape=img0.shape, layout='NTC'),
                mx.io.DataDesc(name='image1', shape=img1.shape, layout='NTC'),
                mx.io.DataDesc(name='qtype', shape=qtype.shape,layout='NTC')
                ],
                         provide_label=[(self.label_name, label.shape)])



numgpus = 8
parser = argparse.ArgumentParser(description="VQA",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--out-dim1', type=int, default=8000,
                    help='max num of epochs')
parser.add_argument('--out-dim2', type=int, default=4000,
                    help='max num of epochs')

parser.add_argument('--num-hidden', type=int, default=2000,
                    help='max num of epochs')

parser.add_argument('--dropout', type=float, default=0.3,
                    help='max num of epochs')
parser.add_argument('--lr', type=float, default= 0.1,
                    help='max num of epochs')

parser.add_argument('--batch-size', type=int, default=100*numgpus,
                    help='the batch size.')
@mx.init.register
class Plusminusone(mx.init.Initializer):
    """Initialize the weight with random +1 or -1 """
    def __init__(self):
        super(Plusminusone, self).__init__()

    def _init_weight(self, _, arr):
        arr[:] = np.random.randint(0, 2, arr.shape)*2-1
        

@mx.init.register
class Index(mx.init.Initializer):
    """Initialize the weight within range [0, up_value]
    Parameters
    ----------
    up_value : int
        The range of the index
    """
    def __init__(self, up_value):
        super(Index, self).__init__(up_value = up_value)
        self.up_value = up_value

    def _init_weight(self, _, arr):
        arr[:] = np.random.randint(0, self.up_value,arr.shape)

def eval_metrics(): 
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [mx.metric.Accuracy(),mx.metric.CrossEntropy()]:
        eval_metrics.add(child_metric)
    return eval_metrics

def evaluation_callback(iter_no, sym, arg, aux):
    
    infile = r'/home/ubuntu/efs/con-att_sgd_no1_repeat_tdiuc_mct-w2v_outdim1%d_outdim2%d_dropout%f_lr%f_adam.log' %(args.out_dim1,args.out_dim2,args.dropout, args.lr)
    with open(infile) as f:
        f = f.readlines()
        val_acc = []
        val_acc_key = ["Validation-accuracy"]
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
            mx.model.save_checkpoint('/home/ubuntu/efs/con-att_sgd_no1_repeat_tdiuc_mct-w2v_outdim1%d_outdim2%d_dropout%f_lr%f_adam' %(args.out_dim1,args.out_dim2,args.dropout,args.lr), iter_no, sym, arg, aux)

def get_gradient(g):
    # get flatten list
    grad = g.asnumpy().flatten()
    # logging using tensorboard, use histogram type.
    return mx.nd.norm(g)/np.sqrt(g.size)


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




def train(args):
    print(args.dropout)
    logging.basicConfig(filename='/home/ubuntu/efs/con-att_sgd_no1_repeat_tdiuc_mct-w2v_outdim1%d_outdim2%d_dropout%f_lr%f_adam.log' %(args.out_dim1,args.out_dim2,args.dropout, args.lr), level=logging.INFO)
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

    
    print('load w2v...')
    model = gensim.models.KeyedVectors.load_word2vec_format('/home/ubuntu/efs/GoogleNews-vectors-negative300.bin.gz', binary=True)
    with open('/home/ubuntu/efs/top1480ans/data_prepro.json') as json_data:
        d = json.load(json_data)
        train_q_w2v = load_data_with_word2vec(train_question,d['ix_to_word'],model)
        val_q_w2v = load_data_with_word2vec(val_question,d['ix_to_word'],model)

    print 'loading images...'
    train_img = np.loadtxt('/home/ubuntu/efs/top1480ans/train_img.txt')
    val_img = np.loadtxt('/home/ubuntu/efs/top1480ans/val_img.txt')
    print 'loading answers...'
    train_ans = np.loadtxt("/home/ubuntu/efs/top1480ans/train_answer_num.txt")
    val_ans = np.loadtxt("/home/ubuntu/efs/top1480ans/val_answer_num.txt")
    print("loading question type...")
    train_qtype = np.load("/home/ubuntu/efs/top1480ans/train_question_type.npz")['arr_0']
    val_qtype = np.load("/home/ubuntu/efs/top1480ans/val_question_type.npz")['arr_0']
    

    
    seq_len = 26
    num_embed = 100
    num_lstm_layer = 2
    num_lstm_hidden = 1024
    num_filter = [512,1]
    workspace = 1024
    ################################################
    layout = 'NT'
    
    vocabulary_size = 7498
    print 'loading rcnn dictionaries'
    val_dict1 = np.load('/home/ubuntu/efs/top1480ans/rcnn-36-2048-val.npy')
    train_dict1 = np.load('/home/ubuntu/efs/top1480ans/rcnn-36-2048-train.npy')
    print 'loading resnet dictionaries'
    val_dict0 = np.load('/home/ubuntu/efs/top1480ans/resnet-2048-14-14-val.npy')
    train_dict0 = np.load('/home/ubuntu/efs/top1480ans/resnet-2048-14-14-train.npy')
    print 'loading finished!'
    data_train  = VQAtrainIterDic(train_dict0, train_dict1, train_img, train_q_w2v,train_question, train_ans, train_qtype, args.batch_size,layout=layout)

    
    np.random.seed(2)
    idx = np.random.choice(range(538543), 1000, replace=False)
    val_img = val_img[idx]
    val_q_w2v = val_q_w2v[idx]
    val_question = val_question[idx]
    val_ans = val_ans[idx]
    val_qtype = val_qtype[idx]
    
    data_eva = VQAtrainIterDic(val_dict0,val_dict1,val_img, val_q_w2v,val_question, val_ans,val_qtype, args.batch_size, layout=layout)


    cell = mx.rnn.SequentialRNNCell()
    for i in range(num_lstm_layer):
        cell.add(mx.rnn.FusedRNNCell(num_lstm_hidden, num_layers=1,
                                     mode='lstm', prefix='lstm_l%d'%i,
                                     bidirectional=False))
    def sym_gen(seq_len):

        lstm_data = mx.sym.Variable('text')
        w2v_data = mx.sym.Variable('w2v')
        label = mx.sym.Variable('softmax_label')
        rcnn_img_data = mx.sym.Variable('image1')
        resnet_img_data = mx.sym.Variable('image0')
        qtype = mx.symbol.Variable('qtype')
        lstm_data = mx.sym.transpose(lstm_data)
        embed = mx.sym.Embedding(data=lstm_data, input_dim=vocabulary_size,output_dim=100,name='embed')
        output, _ = cell.unroll(seq_len, inputs=embed, merge_outputs=True, layout='TNC')
        LSTM = mx.sym.SequenceLast(data = output)
        
        text_data = mx.sym.Concat(LSTM,w2v_data,dim = 1)
        text_data = mx.sym.L2Normalization(data = text_data, name = 'text_l2')
        
        rcnn_image_data = mx.sym.L2Normalization(data = rcnn_img_data, name = 'rcnn_l2')
        rcnn_image_data_reshape = mx.sym.Reshape(rcnn_image_data, shape=(-3,-2),name = 'rs1')   #N*36*2048--> (N*36) * 2048
        
        resnet_image_data = mx.sym.L2Normalization(data = resnet_img_data, name = 'resnet_l2')   #N 2048 14(H) 14
        resnet_image_data_reshape = mx.sym.SwapAxis(resnet_image_data,dim1 = 1,dim2 = 2)   #N 14(H) 2048 14
        resnet_image_data_reshape = mx.sym.SwapAxis(resnet_image_data_reshape,dim1 = 2,dim2 = 3)   #N 14(H) 14 2048        
        ###############################        
        ######### rcnn-mcb ############
        compute_size = args.batch_size/numgpus
        S1 = mx.sym.Variable('s1',init = Plusminusone(),shape = (1,2048))
        H1 = mx.sym.Variable('h1',init = Index(args.out_dim1),shape = (1,2048))
        S2 = mx.sym.Variable('s2',init = Plusminusone(),shape = (1,1024+300))
        H2 = mx.sym.Variable('h2',init = Index(args.out_dim1),shape = (1,1324))

        cs1 = mx.contrib.sym.count_sketch( data = rcnn_image_data_reshape,s=S1, h = H1 ,name='cs1',out_dim = args.out_dim1) 
        cs2 = mx.contrib.sym.count_sketch( data = text_data,s=S2, h = H2 ,name='cs2',out_dim = args.out_dim1) 

        cs1_reshape = mx.sym.Reshape(cs1,(args.batch_size/numgpus,6,6,-1),name = 'rs2')

        cs2_copy = mx.sym.Reshape(cs2,(args.batch_size/numgpus, args.out_dim1,1,1), name = 'rs3')
        cs2_copy2 = mx.sym.broadcast_to(cs2_copy,shape=(0,0,6,6))
        cs2_reshape = mx.sym.SwapAxis(cs2_copy2,dim1 = 1,dim2 = 2)
        cs2_reshape = mx.sym.SwapAxis(cs2_reshape,dim1 = 2,dim2 = 3)

        fft1 = mx.contrib.sym.fft(data = cs1_reshape, name='fft1', compute_size = compute_size) 
        fft2 = mx.contrib.sym.fft(data = cs2_reshape, name='fft2', compute_size = compute_size) 
        c = fft1*fft2
        ifft1 = mx.contrib.sym.ifft(data = c, name='ifft1', compute_size = compute_size) # N 6 6 out-dim
        bn0 = mx.sym.L2Normalization(data = ifft1, name = 'bn0')
        bn0_swapaxis = mx.sym.SwapAxis(bn0,dim1 = 1,dim2 = 3)
        bn0_swapaxis = mx.sym.SwapAxis(bn0_swapaxis,dim1 = 2,dim2 = 3)   # N out-dim 6 6

        cv0 = mx.sym.Convolution(data=bn0_swapaxis, num_filter=num_filter[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                          no_bias=True, name="conv0", workspace=workspace,layout = 'NCHW')
        re0 = mx.sym.Activation(data=cv0, act_type='relu', name='relu0')
        cv1 = mx.sym.Convolution(data=re0, num_filter=num_filter[1], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                          no_bias=True, name="conv1", workspace=workspace,layout = 'NCHW')
        body = mx.symbol.SoftmaxActivation(data=cv1, mode = "channel", name='softmax0')  #N 1 6(H) 6 

        img_swap = mx.sym.SwapAxis(rcnn_image_data,dim1 = 1,dim2 = 2) #N 36 2048    to    N 2048 36
        img_swap = mx.sym.Reshape(img_swap, shape=(args.batch_size/numgpus,2048,6,6),name = 'rs4')   #N 2048 36--> N 2048 6 6

        image_attention = mx.sym.broadcast_mul(img_swap,body)
        image_attention = mx.sym.sum(image_attention,axis = (2,3))   #N 2048

        model0 = mx.sym.L2Normalization(data = image_attention, name = 'rcnn_feature_l2')
        ######################################
        ######resnet-mcb#################
        compute_size = args.batch_size/numgpus
        S5 = mx.sym.Variable('s5',init = Plusminusone(),shape = (1,2048))
        H5 = mx.sym.Variable('h5',init = Index(args.out_dim1),shape = (1,2048))
        
        cs5 = mx.contrib.sym.count_sketch( data = resnet_image_data_reshape,s=S5, h = H5 ,name='cs5',out_dim = args.out_dim1) 

        cs6_copy = mx.sym.broadcast_to(cs2_copy,shape=(0,0,14,14))
        cs6_reshape = mx.sym.SwapAxis(cs6_copy,dim1 = 1,dim2 = 2)
        cs6_reshape = mx.sym.SwapAxis(cs6_reshape,dim1 = 2,dim2 = 3)

        fft5 = mx.contrib.sym.fft(data = cs5, name='fft5', compute_size = compute_size) 
        fft6 = mx.contrib.sym.fft(data = cs6_reshape, name='fft6', compute_size = compute_size) 
        c3 = fft5*fft6
        ifft3 = mx.contrib.sym.ifft(data = c3, name='ifft3', compute_size = compute_size) # N 6 6 out-dim
        bn1 = mx.sym.L2Normalization(data = ifft3, name = 'bn5')
        bn1_swapaxis = mx.sym.SwapAxis(bn1,dim1 = 1,dim2 = 3)
        bn1_swapaxis = mx.sym.SwapAxis(bn1_swapaxis,dim1 = 2,dim2 = 3)   # N out-dim 14 14

        cv2 = mx.sym.Convolution(data=bn1_swapaxis, num_filter=num_filter[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                          no_bias=True, name="conv2", workspace=workspace,layout = 'NCHW')
        re1 = mx.sym.Activation(data=cv2, act_type='relu', name='relu1')
        cv3 = mx.sym.Convolution(data=re1, num_filter=num_filter[1], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                          no_bias=True, name="conv3", workspace=workspace,layout = 'NCHW')
        body2 = mx.symbol.SoftmaxActivation(data=cv3, mode = "channel", name='softmax1')  #N 1 6(H) 6 

        
        image_attention1 = mx.sym.broadcast_mul(resnet_image_data,body2)
        image_attention1 = mx.sym.sum(image_attention1,axis = (2,3))   #N 2048

        model1 = mx.sym.L2Normalization(data = image_attention1, name = 'resnet_feature_l2')
        #################################
        ############# QTA ###############
        mask = mx.sym.FullyConnected(data = qtype, num_hidden=2048*2, name='mask')
        mask = mx.sym.softmax(mask,axis=1)
        concat1 = mx.sym.Concat(model0,model1,dim = 1)
        concat1 = concat1*mask
        concat1 = mx.sym.L2Normalization(data = concat1, name = 'final_representation')
        concat2 = mx.sym.Concat(concat1,text_data,dim = 1)
        #################################
        ############# MLP ###############
        fc1 = mx.sym.FullyConnected(data = concat2, num_hidden=args.num_hidden, no_bias = True)
        act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
        bn1 = mx.sym.BatchNorm(data = act1, name = 'bn3')
        drop = mx.sym.Dropout(bn1,p = args.dropout)
        fc2 = mx.sym.FullyConnected(data = drop, num_hidden=1480)
        #act2 = mx.symbol.Activation(data = fc2, name='sig1', act_type="sigmoid")
        out = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
        return out, ('w2v','text','image0','image1','qtype'), ('softmax_label',)

  
    mod = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = 30,
        context             = [mx.gpu(i) for i in range(numgpus)])
    
        
    data_shapes = [     
           mx.io.DataDesc(name='w2v', shape=(args.batch_size,300 ), layout='NTC'),    
        mx.io.DataDesc(name='text', shape=(args.batch_size,30 ), layout='NTC'),
                mx.io.DataDesc('image0',(args.batch_size,2048,14,14),layout='NTC'),
    mx.io.DataDesc('image1',(args.batch_size,36,2048),layout='NTC'),
    mx.io.DataDesc('qtype',(args.batch_size,12),layout='NTC')]
    label_shapes = [mx.io.DataDesc(
                    'softmax_label',
                    (args.batch_size,),
                    layout='N')]
    mod.bind(data_shapes=data_shapes, label_shapes = label_shapes, for_training=True, grad_req="write")
    mod.init_params()

    print('start training...')
    epoch_size = 1115299/args.batch_size
    
    mod.fit(data_train, data_eva, num_epoch=5000, eval_metric=eval_metrics(), 
            optimizer='Adam', optimizer_params=(('learning_rate', args.lr),),
            epoch_end_callback=evaluation_callback
           )
    

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    
    train(args)

