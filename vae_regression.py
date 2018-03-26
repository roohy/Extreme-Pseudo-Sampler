import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import matplotlib.pyplot as plt 
import sys
from sklearn.decomposition import PCA
from regressor import LogisticRegressor



vindim = None
class VariantionalAutoencoder(object):

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
        f1 = fc(self.x, 15000, scope='enc_fc1', activation_fn=tf.nn.relu)
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
        self.x_hat = fc(g3, self.vindim, scope='dec_fc4', activation_fn=tf.sigmoid)

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
def load_and_normalize(expressions_address,diag_address):
    expressions= pd.read_csv(expressions_address,delimiter=',')
    diags = pd.read_csv(diag_address,delimiter=',')
    temp_con = expressions.values[:,1:]
    temp_con = temp_con.astype(np.float32)
    con  = temp_con.T
    np_diags = np.zeros((con.shape[0],1))
    np_diags[:,0] = [1  if item == 'cancer' else 0 for item in diags.values[:,1]]
    
    min_con = np.min(con,axis=0)
    max_con = np.max(con,axis=0)
    con = con-min_con
    con = con/(max_con-min_con +(1e-10))
    cancer_ones = np_diags[:,0]==1
    cancer_count = np.sum(cancer_ones)
    print("Cancer Count",cancer_count)
    testin = np.random.choice(cancer_count,20)
    trainin = np.array([x for x in range(con.shape[0]) if x not in testin])
    return con,np_diags,cancer_ones,cancer_count,testin,trainin

def create_and_train_vae(con):
    tf.reset_default_graph()
    vindim = con.shape[1]
    model = VariantionalAutoencoder(learning_rate=1e-4, batch_size=100, n_z=500,vindim=vindim)
    print("TRAINING VARIATIONAL:")
    for epoch in range(55):
    
        for iter in range((con.shape[0]) // 20):
            choices = np.random.choice(con.shape[0],20)
            loss = model.run_single_step(con[choices,:])
            #loss = model.run_single_step(X[Y[:,1]==1,:],Y[Y[:,1]==1,:])
        
        if epoch % 5 == 0:
            print('[Epoch {}] Loss: {}'.format( epoch, loss))
    return model

def do_regression(vat,np_diags,trainin,epochs):
    tf.reset_default_graph()
    model = LogisticRegressor(learning_rate=1e-4,input_dim=vat.shape[1])
    print('TRAINIG LATENT REGRESSOR:')
    for epoch in range(epochs):
        for iter in range(len(trainin) // 20):
            choices = np.random.choice(len(trainin),20)
            loss = model.run_single_step(vat[trainin[choices],:],np_diags[trainin[choices],:])
            #loss = model.run_single_step(X[Y[:,1]==1,:],Y[Y[:,1]==1,:])
            
        if epoch % 5 == 0:
            print('[Epoch {}] Loss: {}'.format(epoch, loss))
    labels = model.classifier(vat)
    latent_reg_acc = np.sum(labels[:] == np_diags[:,0])/np_diags.shape[0]
    print('Latent Regressor Accuracy is :',latent_reg_acc)
    return model

if __name__ == '__main__':
    expressions_address = sys.argv[1]
    diag_address = sys.argv[2]
    output_address = sys.argv[3]
    model_address = sys.argv[4]
    print('*****************************************************')
    print(expressions_address)
    print('*****************************************************')
    con,np_diags,cancer_ones,cancer_count,testin,trainin = load_and_normalize(expressions_address,diag_address)


    #VAE FIRST LOAD
    vindim = con.shape[1]
    print('VARIATIONAL INITIATION....')
    model = create_and_train_vae(con)
    saver = tf.train.Saver()
    print('SAVING VAE...')

    saver.save(model.sess,model_address+'_vae.ckpt')
    vat = model.transformer(con)

    #First Regression
    model = do_regression(vat,np_diags,trainin,650)
    print('SAVING REGRESSOR')
    saver = tf.train.Saver()
    saver.save(model.sess,model_address+'latent_reg.ckpt')
    w = model.sess.run(model.W)
    b = model.sess.run(model.b)
    
    dists = vat.dot(w) + b

    max_point = vat[np.argmax(dists),:]
    min_point = vat[np.argmin(dists),:]

    cov = np.eye(500)
    cov = cov*0.2

    max_rand = np.random.multivariate_normal(max_point,cov,200)

    min_rand = np.random.multivariate_normal(min_point,cov,200)
    print('RESTORING VAE...')
    tf.reset_default_graph()
    model = VariantionalAutoencoder(learning_rate=1e-4,batch_size=100, n_z=500)
    saver = tf.train.Saver()
    saver.restore(model.sess,model_address+'_vae.ckpt')
    max_generated = model.generator(max_rand)
    min_generated = model.generator(min_rand)
    '''
    new_data = np.concatenate((con,min_generated,max_generated),axis=0)
    new_mean = np.mean(new_data,axis=0)
    new_var = np.var(new_data,axis=0)
    nn_data = (new_data-new_mean)/(new_var)
    '''

    ex_data = np.concatenate((min_generated,max_generated),axis=0)
    fullbool = np.zeros((400,1))
    fullbool[200:400,0]+=1
    tf.reset_default_graph()
    print("INITIATING EXAGGERATED REGRESSOR...")
    model = LogisticRegressor(learning_rate=1e-4,input_dim=ex_data.shape[1])
    print("training main regressor:")
    for epoch in range(650):
        for iter in range(1000 // 20):
            choices = np.random.choice(400,20)
            loss = model.run_single_step(ex_data[choices,:],fullbool[choices,:])
            
            
        if epoch % 5 == 0:
            print('[Epoch {}] Loss: {}'.format(epoch, loss))
    
    labels = model.classifier(con)
    orig_reg_acc = np.sum(labels)/200
    print("Original Regressor Accuracy is ",orig_reg_acc)
    orig_w = model.sess.run(model.W)
    
    sortedargs = np.argsort(-np.fabs(orig_w[:,0]))
    temp_names = expressions.values[:,0]
    with open(output_address,'w') as outputfile:
        for item in sortedargs :
            outputfile.write(temp_names[item]+'\n')
    print("ALL DONE!!! GO00o0o0d LUCKKKK!!!")
