import mxnet as mx
import numpy as np
import codecs, json
import os, h5py, sys, argparse
import time
import argparse
import bisect
import random
import gensim


class VQAtestIter(mx.io.DataIter):
    def __init__(self, img0, img1, nmt, sentences, batch_size, buckets=[10,20,30], invalid_label=-1,
                 text_name='text-kk', nmt_name = 'nmt',
                  img1_name = 'image1',label0_name='softmax_label', dtype='float32',
                 layout='TNC'):
        super(VQAtestIter, self).__init__()
        if not buckets:
            buckets = [i for i, j in enumerate(np.bincount([len(s) for s in sentences]))
                       if j >= batch_size]
        buckets.sort()

        ndiscard = 0
        self.data = [[] for _ in buckets]
        self.nmt = [[] for _ in buckets]
        self.img0 = [[] for _ in buckets]
        self.img1 = [[] for _ in buckets]
        #self.label0 = [[] for _ in buckets]
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
            
            
        self.data = [np.asarray(i, dtype=dtype) for i in self.data]
        self.nmt = [np.asarray(i, dtype=dtype) for i in self.nmt]
        self.img0 = [np.asarray(i, dtype=dtype) for i in self.img0]
        self.img1 = [np.asarray(i, dtype=dtype) for i in self.img1]
        
        
        
        print("WARNING: discarded %d sentences longer than the largest bucket."%ndiscard)

        self.batch_size = batch_size
        self.buckets = buckets
        self.text_name = text_name
        self.nmt_name = nmt_name
        self.img1_name = img1_name
        #self.label0_name = label0_name
        self.dtype = dtype
        self.invalid_label = invalid_label
        self.nd_text = []
        self.nd_nmt = []
        self.nd_img0 = []
        self.nd_img1 = []
        #self.ndlabel = []
        self.major_axis = layout.find('N')
        self.default_bucket_key = max(buckets)

        if self.major_axis == 0:
            self.provide_data = [(text_name, (batch_size, self.default_bucket_key)),
                                 
                                 (nmt_name, (batch_size, self.default_bucket_key)),
                                 (img0_name, (batch_size, self.default_bucket_key)),
                                 (img1_name, (batch_size, self.default_bucket_key)),
                                (qtype_name, (batch_size, self.default_bucket_key)),
                                 ]
            self.provide_label = [(label_name, (batch_size, ))]
        elif self.major_axis == 1:
            self.provide_data = [mx.io.DataDesc(name='nmt', shape=(300,batch_size), layout='TNC'),
                mx.io.DataDesc(name='text-kk', shape=( 30,batch_size), layout='TNC'),
                mx.io.DataDesc('image1',(2048,batch_size),layout='TNC'),
                mx.io.DataDesc('image0',(2048,batch_size),layout='TNC')
                ]
            
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
        #self.ndlabel0 = []
        for i,buck in enumerate(self.data):
            
            self.nd_text.append(mx.ndarray.array(buck, dtype=self.dtype)) 
            self.nd_nmt.append(mx.ndarray.array(self.nmt[i], dtype=self.dtype))
            self.nd_img0.append(mx.ndarray.array(self.img0[i], dtype=self.dtype))
            self.nd_img1.append(mx.ndarray.array(self.img1[i], dtype=self.dtype))
            #self.ndlabel0.append(mx.ndarray.array(self.label0[i], dtype=self.dtype))
            
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
            #label0 = self.ndlabel0[i][j:j+self.batch_size]
        else:
            nmt = self.nd_nmt[i][j:j + self.batch_size]
            img0 = self.nd_img0[i][j:j + self.batch_size]
            img1 = self.nd_img1[i][j:j + self.batch_size]
            text = self.nd_text[i][j:j + self.batch_size]
            #label = self.ndlabel[i][j:j+self.batch_size]
            qtype = self.nd_qtype[i][j:j+self.batch_size]
        
        data = [  nmt, text, img0, img1]
        return mx.io.DataBatch(data, label = None, pad = 0,
                         bucket_key=self.buckets[i],
                           provide_data=[
                mx.io.DataDesc(name='nmt', shape=nmt.shape,layout='TNC'),
                mx.io.DataDesc(name='text-kk', shape=text.shape, layout='TNC'),
                mx.io.DataDesc(name='image0', shape=img0.shape, layout='TNC'),
                mx.io.DataDesc(name='image1', shape=img1.shape, layout='TNC'),
                ],
                       )






parser = argparse.ArgumentParser(description="VQA",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=256,#317,
                    help='the batch size.')

def eval_metrics(): 
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [mx.metric.Accuracy(),mx.metric.CrossEntropy()]:
        eval_metrics.add(child_metric)
    return eval_metrics
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


####### GLOBAL PARAMETERS ##############
## you can download from https://github.com/VT-vision-lab/VQA_LSTM_CNN Evaluation section ##

def test(args):
    print 'loading dataset...'
    print 'loading lstm questions...'
    input_ques_h5 = '/home/ubuntu/efs/vqa1/data_prepro.h5'
    test_data = {}
    def right_align(seq,lengths):
        v = np.zeros(np.shape(seq))
        N = np.shape(seq)[1]
        for i in range(np.shape(seq)[0]):
            v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
        return v


    with h5py.File(input_ques_h5,'r') as hf:
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)-1
        
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        

        test_data['question'] = right_align(test_data['question'], test_data['length_q'])
        print(max(test_data['length_q']))

    test_question = test_data['question']
    
    
    
    
    model = gensim.models.KeyedVectors.load_word2vec_format('/home/ubuntu/efs/GoogleNews-vectors-negative300.bin.gz', binary=True)
    with open('/home/ubuntu/efs/vqa1/data_prepro.json') as json_data:
        d = json.load(json_data)
        #train_q_w2v = load_data_with_word2vec(train_question,d['ix_to_word'],model)
        test_q_w2v = load_data_with_word2vec(test_question,d['ix_to_word'],model)

    print 'loading rcnn feature...'
    test_img0 = np.load('/home/ubuntu/efs/vqa1/rcnn_test_2048_img_feature.npy')
    
    print 'loading resnet feature...'
    test_img1 = np.load('/home/ubuntu/efs/vqa1/resnet_test_2048_img_feature.npy')
    
   
    data_shapes = [     
           mx.io.DataDesc(name='nmt', shape=(300,args.batch_size ), layout='TNC'),    
        mx.io.DataDesc(name='text-kk', shape=(30,args.batch_size ), layout='TNC'),
                mx.io.DataDesc('image0',(2048,args.batch_size),layout='TNC'),
    mx.io.DataDesc('image1',(2048,args.batch_size),layout='TNC'),
    ]


    print('start')
    layout = 'TN'
    data_test = VQAtestIter(test_img0, test_img1, test_q_w2v, test_question, args.batch_size, layout=layout)
    
    print("loading params")
    sym, arg_params, aux_params = mx.model.load_checkpoint('B-old-hidden4000.000000_hidden6000.000000_drop0.500000_lr0.010000_sgd',96)
    new_args = dict({k:arg_params[k] for k in arg_params}) 

    def sym_gen(seq_len):
        
        sym, arg_params, aux_params = mx.model.load_checkpoint('B-old-hidden4000.000000_hidden6000.000000_drop0.500000_lr0.010000_sgd',96)
        all_layers = sym.get_internals()
        net = all_layers['softmax'+'_output']
        #net = mx.sym.argmax(net,axis=1, keepdims=True)
        return net, ('text-kk','nmt','image0','image1'), ('softmax_label',)
    
    mod = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = 30,
        context             = mx.gpu(3))
    
    mod.bind(for_training=False, data_shapes=data_shapes, label_shapes = None)
    
    mod.set_params(arg_params=new_args, aux_params=aux_params,allow_missing=True)
    ############################################
    y = mod.predict(data_test)
    y = np.argmax(y.asnumpy(), axis = 1)
    print(len(y))
    
    test_q_id = np.loadtxt('/home/ubuntu/efs/vqa1/test_question_id.txt')
    
    file = open("/home/ubuntu/efs/vqa1/data_prepro.json")
    dataset = json.load(file)
    
    result = []
    print(len(y))
    for i in range (0,len(y)):
        ans = dataset['ix_to_ans'][str(y[i]+1)]
        result.append({u'answer': ans, u'question_id': str(test_q_id[i])})

    # Save to JSON
    print 'Saving result...'
    my_list = list(result)
    dd = json.dump(my_list,open('test_result.json','w'))

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    
    test(args)
