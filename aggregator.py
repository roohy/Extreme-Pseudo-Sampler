import numpy as np
import glob
import sys
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
from vae_regression import VariantionalAutoencoder
from regressor import LogisticRegressor

class VariantionalAutoencoder2(object):

    def __init__(self, learning_rate=1e-3, batch_size=100, n_z=10,vindim=100):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        self.vindim = vindim
        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, self.vindim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f0 = fc(self.x, 30000, scope='enc_fc0', activation_fn=tf.nn.relu)
        f1 = fc(f0, 15000, scope='enc_fc1', activation_fn=tf.nn.relu)
        f2 = fc(f1, 10000, scope='enc_fc2', activation_fn=tf.nn.relu)
        f3 = fc(f2, 2000, scope='enc_fc3', activation_fn=tf.nn.relu)
        #f3 = fc(f3, 500, scope='enc_fc3', activation_fn=tf.nn.elu)
        self.z_mu = fc(f3, self.n_z, scope='enc_fc4_mu', activation_fn=None)
        self.z_log_sigma_sq = fc(f3, self.n_z, scope='enc_fc4_sigma', activation_fn=None)
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq),
                               mean=0, stddev=1, dtype=tf.float32)
        #zzz = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps
        #self.zzz = tf.Print(zzz,[zzz], message="my Z-values:")
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps
        
        # Decode
        # z -> x_hat
        g1 = fc(self.z, 2000, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(g1, 10000, scope='dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, 15000, scope='dec_fc3', activation_fn=tf.nn.relu)
        g4 = fc(g3, 30000, scope='dec_fc4', activation_fn=tf.nn.relu)
        self.x_hat = fc(g4, self.vindim, scope='dec_fc5', activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-9
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+ (1-self.x_hat)), 
            axis=1
        )
        #recon_loss = tf.reduce_sum((self.x_hat-self.x)**2,axis=1)

        #recon_loss = tf.nn.l2_loss(self.x_hat-self.x)
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # Kullback Leibler divergence: measure the difference between two distributions
        # Here we measure the divergence between the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = tf.reduce_mean(latent_loss + recon_loss)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        #self.train_op = tf.train.AdamOptimizer(
        #    learning_rate=self.learning_rate).minimize(self.total_loss)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, loss, recon_loss, latent_loss = self.sess.run(
            [self.train_op, self.total_loss, self.recon_loss, self.latent_loss],
            feed_dict={self.x: x}
        )

        return loss, recon_loss, latent_loss

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat
    
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z


class BatchMaker(object):
    def __init__(self):
        self.batch_number = -1
    def load_data(self,data):
        self.data = data
        self.shuffle()
    
    def shuffle(self):
        self.current_q = np.random.permutation(self.data.shape[0])
        self.head = 0
        self.batch_number += 1
    def get_batch(self,batch_size):
        if self.current_q.shape[0]-self.head >= batch_size:
            result = self.data[self.current_q[self.head:self.head+batch_size],:]
            self.head += batch_size
            return result
        else:
            if self.head == self.current_q.shape[0]:
                self.shuffle()
                self.head = batch_size
                return self.data[self.current_q[:batch_size],:]
            else:
                result = self.data[self.current_q[self.head:],:]
                self.shuffle()
                return result
        return None

def do_regression(vat,np_diags,epochs):
    tf.reset_default_graph()
    model = LogisticRegressor(learning_rate=1e-4,input_dim=vat.shape[1])
    print('TRAINIG LATENT REGRESSOR:')
    bm = BatchMaker()
    bm.load_data(np.concatenate([vat,np_diags],axis=1))
    
    while bm.batch_number < epochs:
        
        batch = bm.get_batch(100)
        loss = model.run_single_step(batch[:,:vat.shape[1]],batch[:,-1:])
            
        if bm.batch_number % 5 == 0:
            print('[Epoch {}] Loss: {}'.format(bm.batch_number, loss))
    labels = model.classifier(vat)
    latent_reg_acc = np.sum(labels[:] == np_diags[:,0])/np_diags.shape[0]
    print('Latent Regressor Accuracy is :',latent_reg_acc)
    return model

results = []
counts = {}
total_count = 0
feature_count = 56963

list_of_sample_directories = glob.glob('./'+'TCGA_*/')

for directory in list_of_sample_directories:
    all_samples = glob.glob(directory+'*TCGA*.csv')
    counts[directory] = len(all_samples)
    total_count += len(all_samples)


counter = 0 
sample_mat = np.zeros((total_count,feature_count))
all_names = []
for directory in list_of_sample_directories:
    all_samples = glob.glob(directory+'*TCGA*.csv')
    for sample in all_samples:
        all_names.append(sample)
        with open(sample,'r') as sample_file:
            line_counter = 0
            for line in sample_file:
                sample_mat[counter,line_counter] = line.strip().split('\t')[1]
                line_counter += 1 
        counter += 1

min_val = np.min(sample_mat,axis=0)
max_val = np.max(sample_mat,axis=0)

new_mat = sample_mat[:,~(max_val-min_val == 0)]

np.save('invalids.npy',max_val-min_val == 0)

min_val = np.min(new_mat,axis=0)
max_val = np.max(new_mat,axis=0)


normalized_mat = (new_mat-min_val)/(max_val-min_val)

tf.reset_default_graph()
vindim = 55682#normalized_mat.shape[1]
model = VariantionalAutoencoder2(learning_rate=1e-4, batch_size=100, n_z=500,vindim=vindim)
dataset = BatchMaker()
dataset.load_data(normalized_mat)

counter = 0
loss_list = []
while dataset.batch_number < 10:
    loss = model.run_single_step(dataset.get_batch(20))
    counter += 1 
    if counter % 5 == 0 :
        print(loss) 
    loss_list.append(loss)

saver = tf.train.Saver()
saver.save(model.sess,'vae_all_rms.ckpt')
transformed = model.transformer(normalized_mat)

diags = np.zeros((transformed.shape[0],1))
counter = 0 
for directory in list_of_sample_directories:
    all_samples = glob.glob(directory+'*TCGA*.csv')
    for sample in all_samples:
        if 'cancer' in sample:
            diags[counter] = 1
        elif 'normal' in sample:
            diags[counter] = 0
        else:
            print("ERRROOORRR")
        counter += 1




head_counter = 0
list_of_models = []
list_of_ex_sample = []
for item in list_of_sample_directories:
    sample_count = counts[item]
    vat = transformed[head_counter:head_counter+sample_count]
    
    model = do_regression(vat,diags[head_counter:head_counter+sample_count,:],550)
    w = model.sess.run(model.W)
    b = model.sess.run(model.b)
    
    dists = vat.dot(w) + b

    max_point = vat[np.argmax(dists),:]
    min_point = vat[np.argmin(dists),:]

    cov = np.eye(500)
    cov = cov*0.2

    max_rand = np.random.multivariate_normal(max_point,cov,200)

    min_rand = np.random.multivariate_normal(min_point,cov,200)
    list_of_models.append((w,b))
    list_of_ex_sample.append((max_rand,min_rand))


list_of_ex_real = []
for item in list_of_ex_sample:
    list_of_ex_real.append(np.concatenate([model.generator(item[0]),model.generator(item[1])],axis=0))



tf.reset_default_graph()
vindim = 55682#normalized_mat.shape[1]
model = VariantionalAutoencoder2(learning_rate=1e-4, batch_size=100, n_z=500,vindim=vindim)

saver = tf.train.Saver()
temp_mod = saver.restore(model.sess,'vae_all_rms.ckpt')

dummy = np.zeros((400,1))
dummy[:200]+=1

list_of_ws = []
for item in list_of_ex_real:
    
    new_item = (item-np.mean(item,axis=0))/np.std(item,axis=0)
    print(new_item.shape)
    model = do_regression(new_item,dummy,550)
    w = model.sess.run(model.W)
    b = model.sess.run(model.b)
    list_of_ws.append(w)

invalid = np.load('invalids.npy')
names = np.load('names.npy')
final_names = names[~invalid]


for i in range(len(list_of_ws)):
    sortedargs = np.argsort(-np.fabs(list_of_ws[i][:,0]))
    with open('RANKINGS_'+list_of_sample_directories[i][7:-1]+'_NORM_20_batch','w') as outputfile:
        for item in sortedargs :
            outputfile.write(final_names[item]+'\n')


