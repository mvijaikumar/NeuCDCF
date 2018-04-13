import tensorflow as tf
import numpy as np
from GCMF import GCMF
from MLP import MLP
from SDAE import SDAE
from GCMF_MLP import GCMF_MLP
from GCMF_SDAE import GCMF_SDAE
from GCMF_MLP_SDAE import GCMF_MLP_SDAE

class NeuCDCF(object):
    def __init__(self,params):
        print ("CHANGE KEY: valid clip")
        self.num_users        = params.num_users
        self.num_items        = params.num_items
        self.method           = params.method
        self.reg_bias         = params.reg_bias
        self.params           = params
        self.mu               = params.mu
        self.mu_source        = params.mu_source
        self.mu_target        = params.mu_target
        self.initializer      = params.initializer
        #self.valid_clip       = params.valid_clip
        self.rating_min_val   = params.rating_min_val
        self.rating_max_val   = params.rating_max_val
        
        
    def define_model(self):
        self.user_indices       = tf.placeholder(tf.int32,   shape=[None],name='user_indices')
        self.user_cdcf_indices  = tf.placeholder(tf.int32,   shape=[None],name='user_cdcf_indices')
        self.item_indices       = tf.placeholder(tf.int32,   shape=[None],name='item_indices')
        self.true_rating        = tf.placeholder(tf.float32, shape=[None],name='true_ratings')
        self.keep_prob          = tf.placeholder(tf.float32, name='keep_prob') 
        self.keep_prob_layer    = tf.placeholder(tf.float32, name='keep_prob_layer') 
        self.domain_indices     = tf.placeholder(tf.int32,   shape=[None],name='domain_indices')
        self.valid_clip        = tf.placeholder(tf.float32,   name='valid_clip')
        #self.valid_clip        = tf.placeholder_with_default(input, shape, name=None)
        
            
        self.mu_embeddings      = tf.Variable(tf.constant([self.mu_target,self.mu_source] 
                                                ,dtype=tf.float32),
                                      dtype=tf.float32,name='mu_embedding',trainable=False)        
        self.user_bias_embeddings = tf.Variable(self.initializer(shape=[self.num_users]),
                                          dtype=tf.float32,name='user_bias_embedding') 
        #self.user_bias_cdcf_embeddings = tf.Variable(self.initializer(shape=[2*self.num_users]),
        #                                  dtype=tf.float32,name='user_bias_embedding') 
        self.user_bias_cdcf_embeddings = tf.Variable(tf.zeros(shape=[2*self.num_users]),
                                          dtype=tf.float32,name='user_bias_embedding') 
        self.item_bias_embeddings = tf.Variable(self.initializer(shape=[self.num_items]),
                                          dtype=tf.float32,name='item_bias_embedding')        
        self.ubias_embeds = tf.nn.embedding_lookup(self.user_bias_embeddings, self.user_indices)
        self.ubias_cdcf_embeds = tf.nn.embedding_lookup(self.user_bias_cdcf_embeddings, 
                                                        self.user_cdcf_indices)
        self.ibias_embeds = tf.nn.embedding_lookup(self.item_bias_embeddings, self.item_indices)   
        self.mu_embeds =  tf.nn.embedding_lookup(self.mu_embeddings, self.domain_indices)
        
        # create gmf object and define gmf model
        if self.method == 'gcmf':
            self.gcmf_model  = GCMF(self.params)
            self.gcmf_model.define_model(self.user_indices,self.user_cdcf_indices,
                                        self.item_indices,self.true_rating,
                                        self.keep_prob,self.keep_prob_layer,self.domain_indices)
            self.pred_rating_model = self.gcmf_model.pred_rating
            
        elif self.method == 'mlp':
            self.mlp_model  = MLP(self.params)
            self.mlp_model.define_model(self.user_indices,self.user_cdcf_indices,
                                        self.item_indices,self.true_rating,
                                        self.keep_prob,self.keep_prob_layer,self.domain_indices)
            self.pred_rating_model = self.mlp_model.pred_rating
        elif self.method == 'sdae':
            self.sdae_model  = SDAE(self.params)
            self.sdae_model.define_model(self.user_indices,self.user_cdcf_indices,
                                        self.item_indices,self.true_rating,
                                         self.keep_prob,self.keep_prob_layer,self.domain_indices)
            self.pred_rating_model = self.sdae_model.pred_rating
        elif self.method == 'gcmf_mlp':
            self.gcmf_mlp_model  = GCMF_MLP(self.params)
            self.gcmf_mlp_model.define_model(self.user_indices,self.user_cdcf_indices,
                                        self.item_indices,self.true_rating,
                                        self.keep_prob,self.keep_prob_layer,self.domain_indices)
            self.pred_rating_model = self.gcmf_mlp_model.pred_rating
        elif self.method == 'gcmf_sdae':
            self.gcmf_sdae_model  = GCMF_SDAE(self.params)
            self.gcmf_sdae_model.define_model(self.user_indices,self.user_cdcf_indices,
                                        self.item_indices,self.true_rating,
                                        self.keep_prob,self.keep_prob_layer,self.domain_indices)
            self.pred_rating_model= self.gcmf_sdae_model.pred_rating
        elif self.method == 'gcmf_mlp_sdae':
            self.gcmf_mlp_sdae_model  = GCMF_MLP_SDAE(self.params)
            self.gcmf_mlp_sdae_model.define_model(self.user_indices,self.user_cdcf_indices,
                                        self.item_indices,self.true_rating,
                                        self.keep_prob,self.keep_prob_layer,self.domain_indices)
            self.pred_rating_model = self.gcmf_mlp_sdae_model.pred_rating
        #====    
        '''
        self.pred_rating    = (self.pred_rating_model +
                               self.ubias_embeds + self.ubias_cdcf_embeds +
                               self.ibias_embeds +self.mu_embeds)
        '''
        self.pred_rating    = (self.pred_rating_model)
        self.pred_rating    = ((1- self.valid_clip) * self.pred_rating + 
                                   self.valid_clip * tf.clip_by_value(self.pred_rating,
                                                 self.rating_min_val,self.rating_max_val))
        print('PRED_RATING : with bias')
        
    def define_loss(self,loss_type='all'): # no_reg ##
        self.recon_error = tf.constant(0.0,dtype=tf.float32)
        if self.method == 'gcmf':
            self.gcmf_model.define_loss(loss_type=loss_type)
            self.regularization_loss = self.gcmf_model.regularization_loss
        elif self.method == 'mlp':
            self.mlp_model.define_loss(loss_type=loss_type)
            self.regularization_loss = self.mlp_model.regularization_loss
        elif self.method =='sdae':
            self.sdae_model.define_loss(loss_type=loss_type)
            self.regularization_loss = self.sdae_model.regularization_loss
            self.recon_error += self.sdae_model.recon_error
        elif self.method =='gcmf_mlp':
            self.gcmf_mlp_model.define_loss(loss_type=loss_type)
            self.regularization_loss = self.gcmf_mlp_model.regularization_loss
        elif self.method =='gcmf_sdae':
            self.gcmf_sdae_model.define_loss(loss_type=loss_type)
            self.regularization_loss = self.gcmf_sdae_model.regularization_loss
            self.recon_error += self.gcmf_sdae_model.recon_error
        elif self.method =='gcmf_mlp_sdae':
            self.gcmf_mlp_sdae_model.define_loss(loss_type=loss_type)
            self.regularization_loss = self.gcmf_mlp_sdae_model.regularization_loss
            self.recon_error += self.gcmf_mlp_sdae_model.recon_error
            
        self.mse_loss = tf.losses.mean_squared_error(self.true_rating,self.pred_rating)
        self.mae_loss = tf.losses.absolute_difference(self.true_rating,self.pred_rating)
        self.se_loss  = tf.reduce_sum(tf.squared_difference(self.true_rating,self.pred_rating))

        self.regularization_loss = (self.regularization_loss + self.reg_bias *
                                     (tf.nn.l2_loss(self.ubias_embeds) +
                                     tf.nn.l2_loss(self.ubias_cdcf_embeds)+
                                     tf.nn.l2_loss(self.ibias_embeds)))
        
        #if loss_type == 'all':
        self.loss = self.mse_loss + self.regularization_loss + self.recon_error
        
# ===============================================================================================#
