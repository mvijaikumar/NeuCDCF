import tensorflow as tf
import subprocess
class Pretrain(object):
    '''Only saves gcmf, mlp, sdae''' '''and neumf_gmf'''
    def __init__(self,params,sess):
        self.sess          = sess
        self.pretrain_path = params.pretrain_path
        self.method        = params.method
        self.pretrain_save = params.pretrain_save
        self.pretrain_load = params.pretrain_load
        
        print "Pretrain: ",self.pretrain_path,self.method,self.pretrain_save,self.pretrain_load
    def save(self,model):
        if self.method == 'gcmf':
            self.save_gcmf(model)
        elif self.method == 'mlp':
            self.save_mlp(model)
        elif self.method == 'sdae':
            self.save_sdae(model)            
        elif self.method == 'neumf_gmf':
            self.save_neumf_gmf(model)
    def load(self,model):
        if self.method == 'gcmf':
            self.load_gcmf(model)        
        elif self.method == 'mlp':
            self.load_mlp(model)
        elif self.method == 'sdae':
            self.load_sdae(model)
        elif self.method == 'gcmf_mlp':
            self.load_gcmf(model.gcmf_mlp_model)
            self.load_mlp(model.gcmf_mlp_model)
        elif self.method == 'gcmf_sdae':
            self.load_gcmf(model.gcmf_sdae_model)
            self.load_sdae(model.gcmf_sdae_model)
        elif self.method == 'gcmf_mlp_sdae':
            self.load_gcmf(model.gcmf_mlp_sdae_model.gcmf_mlp_model)
            self.load_mlp(model.gcmf_mlp_sdae_model.gcmf_mlp_model)
            self.load_sdae(model.gcmf_mlp_sdae_model)
        elif self.method == 'neumf_gmf':
            self.load_neumf_gmf(model)
        
    def save_gcmf(self,model):
        saver = tf.train.Saver({'user_embeddings':model.gcmf_model.user_embeddings,
                                'item_embeddings':model.gcmf_model.item_embeddings,
                                'user_cdcf_embeddings':model.gcmf_model.user_cdcf_embeddings,
                                'W_G':model.gcmf_model.W_G})
        subprocess.call(['mkdir','-p',self.pretrain_path + '/gcmf/'])
        saver.save(self.sess,self.pretrain_path + '/gcmf/gcmf')
        print('Pretained weights are saved at: ' + self.pretrain_path + '/gcmf/gcmf')
        
    def save_mlp(self,model):
        saver = tf.train.Saver({'user_embeddings':model.mlp_model.user_embeddings,
                                'item_embeddings':model.mlp_model.item_embeddings,
                                'user_cdcf_embeddings':model.mlp_model.user_cdcf_embeddings,
                                'W_1':model.mlp_model.W_1,
                                'W_2':model.mlp_model.W_2,
                                'W_MLP':model.mlp_model.W_MLP})
        subprocess.call(['mkdir','-p',self.pretrain_path + '/mlp/'])
        saver.save(self.sess,self.pretrain_path + '/mlp/mlp')
        print('Pretained weights are saved at: ' + self.pretrain_path + '/mlp/mlp')
        
    def save_sdae(self,model):
        saver = tf.train.Saver({'item_embeddings':model.sdae_model.item_embeddings,
                                       'Weight_u1':model.sdae_model.Weight_u1,
                                       'Weight_u2':model.sdae_model.Weight_u2,
                                       'Weight_u3':model.sdae_model.Weight_u3,
                                       'Weight_u4':model.sdae_model.Weight_u4,
                                       'Weight_v1':model.sdae_model.Weight_v1,
                                       'Weight_v2':model.sdae_model.Weight_v2,
                                        'b_u1':model.sdae_model.b_u1,
                                        'b_u2':model.sdae_model.b_u2,
                                        'b_u3':model.sdae_model.b_u3,
                                        'b_u4':model.sdae_model.b_u4,
                                        'b_v1':model.sdae_model.b_v1,
                                        'b_v2':model.sdae_model.b_v2,
                                       'W_AE':model.sdae_model.W_AE})
        subprocess.call(['mkdir','-p',self.pretrain_path + '/sdae/'])
        saver.save(self.sess,self.pretrain_path+'/sdae/sdae')
        print('Pretained weights are saved at: ' + self.pretrain_path + '/sdae/sdae')
    
    
    def save_neumf_gmf(self,model):
        saver = tf.train.Saver({'user_embeddings':model.neumf_gmf_model.user_embeddings,
                                'item_embeddings':model.neumf_gmf_model.item_embeddings,
                                'W_G':model.neumf_gmf_model.W_G})
        subprocess.call(['mkdir','-p',self.pretrain_path + '/neumf_gmf/'])
        saver.save(self.sess,self.pretrain_path + '/neumf_gmf/neumf_gmf')
        print('Pretained weights are saved at: ' + self.pretrain_path + '/neumf_gmf/neumf_gmf')
    # load ==================#
        
    def load_gcmf(self,model):
        saver = tf.train.Saver({'user_embeddings':model.gcmf_model.user_embeddings,
                                'item_embeddings':model.gcmf_model.item_embeddings,
                                'user_cdcf_embeddings':model.gcmf_model.user_cdcf_embeddings,
                                'W_G':model.gcmf_model.W_G})
        saver.restore(self.sess, self.pretrain_path+'/gcmf/gcmf')
        print('Pretained weights are loaded from: ' + self.pretrain_path + '/gcmf/gcmf')
        
    def load_mlp(self,model):
        saver = tf.train.Saver({'user_embeddings':model.mlp_model.user_embeddings,
                                'item_embeddings':model.mlp_model.item_embeddings,
                                'user_cdcf_embeddings':model.mlp_model.user_cdcf_embeddings,
                                'W_1':model.mlp_model.W_1,
                                'W_2':model.mlp_model.W_2,
                                'W_MLP':model.mlp_model.W_MLP})
        saver.restore(self.sess, self.pretrain_path+'/mlp/mlp')
        print('Pretained weights are loaded from: ' + self.pretrain_path + '/mlp/mlp')
        
    def load_sdae(self,model):        
        saver = tf.train.Saver({'item_embeddings':model.sdae_model.item_embeddings,
                                       'Weight_u1':model.sdae_model.Weight_u1,
                                       'Weight_u2':model.sdae_model.Weight_u2,
                                       'Weight_u3':model.sdae_model.Weight_u3,
                                       'Weight_u4':model.sdae_model.Weight_u4,
                                       'Weight_v1':model.sdae_model.Weight_v1,
                                       'Weight_v2':model.sdae_model.Weight_v2,
                                        'b_u1':model.sdae_model.b_u1,
                                        'b_u2':model.sdae_model.b_u2,
                                        'b_u3':model.sdae_model.b_u3,
                                        'b_u4':model.sdae_model.b_u4,
                                        'b_v1':model.sdae_model.b_v1,
                                        'b_v2':model.sdae_model.b_v2,
                                       'W_AE':model.sdae_model.W_AE})
        saver.restore(self.sess,self.pretrain_path+'/sdae/sdae')
        print('Pretained weights are loaded from: ' + self.pretrain_path + '/sdae/sdae')
                
    def load_neumf_gmf(self,model):
        saver = tf.train.Saver({'user_embeddings':model.neumf_gmf_model.user_embeddings,
                                'item_embeddings':model.neumf_gmf_model.item_embeddings,
                                'W_G':model.neumf_gmf_model.W_G})
        saver.restore(self.sess,self.pretrain_path + '/neumf_gmf/neumf_gmf')
        print('Pretained weights are loaded from: ' + self.pretrain_path + '/neumf_gmf/neumf_gmf')