import tensorflow as tf
import subprocess
class Pretrain(object):
    '''Only saves gcmf, mlp, sed''' '''and neumf_gmf'''
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
        elif self.method == 'sed':
            self.save_sed(model)            
        elif self.method == 'neumf_gmf':
            self.save_neumf_gmf(model)
    def load(self,model):
        if self.method == 'gcmf':
            self.load_gcmf(model)        
        elif self.method == 'mlp':
            self.load_mlp(model)
        elif self.method == 'sed':
            self.load_sed(model)
        elif self.method == 'gcmf_mlp':
            self.load_gcmf(model.gcmf_mlp_model)
            self.load_mlp(model.gcmf_mlp_model)
        elif self.method == 'gcmf_sed':
            self.load_gcmf(model.gcmf_sed_model)
            self.load_sed(model.gcmf_sed_model)
        elif self.method == 'gcmf_mlp_sed':
            self.load_gcmf(model.gcmf_mlp_sed_model.gcmf_mlp_model)
            self.load_mlp(model.gcmf_mlp_sed_model.gcmf_mlp_model)
            self.load_sed(model.gcmf_mlp_sed_model)
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
        
    def save_sed(self,model):
        saver = tf.train.Saver({'item_embeddings':model.sed_model.item_embeddings,
                                       'Weight_u1':model.sed_model.Weight_u1,
                                       'Weight_u2':model.sed_model.Weight_u2,
                                       'Weight_u3':model.sed_model.Weight_u3,
                                       'Weight_u4':model.sed_model.Weight_u4,
                                       'Weight_v1':model.sed_model.Weight_v1,
                                       'Weight_v2':model.sed_model.Weight_v2,
                                        'b_u1':model.sed_model.b_u1,
                                        'b_u2':model.sed_model.b_u2,
                                        'b_u3':model.sed_model.b_u3,
                                        'b_u4':model.sed_model.b_u4,
                                        'b_v1':model.sed_model.b_v1,
                                        'b_v2':model.sed_model.b_v2,
                                       'W_AE':model.sed_model.W_AE})
        subprocess.call(['mkdir','-p',self.pretrain_path + '/sed/'])
        saver.save(self.sess,self.pretrain_path+'/sed/sed')
        print('Pretained weights are saved at: ' + self.pretrain_path + '/sed/sed')
    
    
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
        
    def load_sed(self,model):        
        saver = tf.train.Saver({'item_embeddings':model.sed_model.item_embeddings,
                                       'Weight_u1':model.sed_model.Weight_u1,
                                       'Weight_u2':model.sed_model.Weight_u2,
                                       'Weight_u3':model.sed_model.Weight_u3,
                                       'Weight_u4':model.sed_model.Weight_u4,
                                       'Weight_v1':model.sed_model.Weight_v1,
                                       'Weight_v2':model.sed_model.Weight_v2,
                                        'b_u1':model.sed_model.b_u1,
                                        'b_u2':model.sed_model.b_u2,
                                        'b_u3':model.sed_model.b_u3,
                                        'b_u4':model.sed_model.b_u4,
                                        'b_v1':model.sed_model.b_v1,
                                        'b_v2':model.sed_model.b_v2,
                                       'W_AE':model.sed_model.W_AE})
        saver.restore(self.sess,self.pretrain_path+'/sed/sed')
        print('Pretained weights are loaded from: ' + self.pretrain_path + '/sed/sed')
                
    def load_neumf_gmf(self,model):
        saver = tf.train.Saver({'user_embeddings':model.neumf_gmf_model.user_embeddings,
                                'item_embeddings':model.neumf_gmf_model.item_embeddings,
                                'W_G':model.neumf_gmf_model.W_G})
        saver.restore(self.sess,self.pretrain_path + '/neumf_gmf/neumf_gmf')
        print('Pretained weights are loaded from: ' + self.pretrain_path + '/neumf_gmf/neumf_gmf')
