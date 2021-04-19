from PIL import Image
import tensorflow as tf
import tflib
import tflib.ops
import tflib.network
from tqdm import tqdm
import numpy as np
import data_loaders
import time
import os

tf.compat.v1.disable_eager_execution()

BATCH_SIZE      = 20
EMB_DIM         = 80
ENC_DIM         = 256
DEC_DIM         = ENC_DIM*2
NUM_FEATS_START = 64
D               = NUM_FEATS_START*8
V               = 502
NB_EPOCHS       = 50
H               = 20
W               = 50

# with tf.device("/cpu:0"):
#     custom_runner = data_loaders.CustomRunner()
#     X,seqs,mask,reset = custom_runner.get_inputs()
#
# print X,seqs
X = tf.placeholder(shape=(None,None,None,None),dtype=tf.float32)
mask = tf.placeholder(shape=(None,None),dtype=tf.int32)
seqs = tf.placeholder(shape=(None,None),dtype=tf.int32)
learn_rate = tf.placeholder(tf.float32)
input_seqs = seqs[:,:-1]
target_seqs = seqs[:,1:]
emb_seqs = tflib.ops.Embedding('Embedding',V,EMB_DIM,input_seqs)

ctx = tflib.network.im2latex_cnn(X,NUM_FEATS_START,True)
out,state = tflib.ops.im2latexAttention('AttLSTM',emb_seqs,ctx,EMB_DIM,ENC_DIM,DEC_DIM,D,H,W)
logits = tflib.ops.Linear('MLP.1',out,DEC_DIM,V)
predictions = tf.argmax(tf.nn.softmax(logits[:,-1]),axis=1)


loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=tf.reshape(logits,[-1,V]),
    labels=tf.reshape(seqs[:,1:],[-1])
    ), [tf.shape(X)[0], -1])

mask_mult = tf.to_float(mask[:,1:])
loss = tf.reduce_sum(loss*mask_mult)/tf.reduce_sum(mask_mult)

#train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learn_rate)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs)

def predict(set='test',batch_size=1,visualize=True):
    if visualize:
        assert (batch_size==1), "Batch size should be 1 for visualize mode"
    import random
    # f = np.load('train_list_buckets.npy').tolist()
    f = np.load(set+'_buckets.npy', allow_pickle=True).tolist()
    random_key = random.choice(f.keys())
    print(random_key)
    f = f[random_key]
    print(f)
    imgs = []
    print "Image shape: ",random_key
    while len(imgs)!=batch_size:
        start = np.random.randint(0,len(f),1)[0]
        print(f[start][0])
        # imgs.append(np.asarray(Image.open('/content/im2latex-tensorflow/images_processed/100bac2b03.png').convert('YCbCr'))[:,:,0][:,:,None])
        # imgs.append(np.asarray(Image.open('/content/im2latex-tensorflow/images_processed/10413ebe59.png').convert('YCbCr'))[:,:,0][:,:,None])
        # imgs.append(np.asarray(Image.open('/content/im2latex-tensorflow/images_processed/10442b0b73.png').convert('YCbCr'))[:,:,0][:,:,None])
        if os.path.exists('./images_processed/'+f[start][0]):
            imgs.append(np.asarray(Image.open('./images_processed/'+f[start][0]).convert('YCbCr'))[:,:,0][:,:,None])

        # if os.path.exists('/content/im2latex-tensorflow/images_processed/10442b0b73.png'):
        #     imgs.append(np.asarray(Image.open('/content/im2latex-tensorflow/images_processed/10442b0b73.png').convert('YCbCr'))[:,:,0][:,:,None])

    imgs = np.asarray(imgs,dtype=np.float32).transpose(0,3,1,2)
    inp_seqs = np.zeros((batch_size,160)).astype('int32')
    print imgs.shape
    inp_seqs[:,0] = np.load('properties.npy', allow_pickle=True).tolist()['char_to_idx']['#START']
    tflib.ops.ctx_vector = []

    l_size = random_key[0]*2
    r_size = random_key[1]*2
    inp_image = Image.fromarray(imgs[0][0]).resize((l_size,r_size))
    l = int(np.ceil(random_key[1]/8.))
    r = int(np.ceil(random_key[0]/8.))
    properties = np.load('properties.npy', allow_pickle=True).tolist()
    idx_to_chars = lambda Y: ' '.join(map(lambda x: properties['idx_to_char'][x],Y))

    for i in xrange(1,160):
        inp_seqs[:,i] = sess.run(predictions,feed_dict={X:imgs,input_seqs:inp_seqs[:,:i]})
        #print i,inp_seqs[:,i]
        if visualize==True:
            att = sorted(list(enumerate(tflib.ops.ctx_vector[-1].flatten())),key=lambda tup:tup[1],reverse=True)
            idxs,att = zip(*att)
            j=1
            while sum(att[:j])<0.9:
                j+=1
            positions = idxs[:j]
            print "Attention weights: ",att[:j]
            positions = [(pos/r,pos%r) for pos in positions]
            outarray = np.ones((l,r))*255.
            for loc in positions:
                outarray[loc] = 0.
            out_image = Image.fromarray(outarray).resize((l_size,r_size),Image.NEAREST)
            print "Latex sequence: ",idx_to_chars(inp_seqs[0,:i])
            outp = Image.blend(inp_image.convert('RGBA'),out_image.convert('RGBA'),0.5)
            outp.show(title=properties['idx_to_char'][inp_seqs[0,i]])
            # raw_input()
            time.sleep(3)
            os.system('pkill display')

    np.save('pred_imgs',imgs)
    np.save('pred_latex',inp_seqs)
    print "Saved npy files! Use Predict.ipynb to view results"
    return inp_seqs

def score(set='valid',batch_size=32):
    score_itr = data_loaders.data_iterator(set,batch_size)
    losses = []
    start = time.time()
    for score_imgs,score_seqs,score_mask in score_itr:
        _loss = sess.run(loss,feed_dict={X:score_imgs,seqs:score_seqs,mask:score_mask})
        losses.append(_loss)
        print _loss

    set_loss = np.mean(losses)
    perp = np.mean(map(lambda x: np.power(np.e,x), losses))
    print "\tMean %s Loss: ", set_loss
    print "\tTotal %s Time: ", time.time()-start
    print "\tMean %s Perplexity: ", perp
    return set_loss, perp

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
init = tf.global_variables_initializer()
# init = tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./'))
# saver.restore(sess,'./weights_best.ckpt')
## start the tensorflow QueueRunner's
# tf.train.start_queue_runners(sess=sess)
## start our custom queue runner's threads
# custom_runner.start_threads(sess)

losses = []
times = []
print "Compiled Train function!"
## Test is train func runs
# train_fn(np.random.randn(32,1,128,256).astype('float32'),np.random.randint(0,107,(32,50)).astype('int32'),np.random.randint(0,2,(32,50)).astype('int32'), np.zeros((32,1024)).astype('float32'))
i=0
lr = 0.1
best_perp = np.finfo(np.float32).max

predict(set='test',batch_size=1, visualize=False)