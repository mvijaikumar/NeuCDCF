import tensorflow as tf
import numpy as np

class GCMF(object):
    def __init__(self,params):
        
        self.num_factors      = params.num_factors        
        self.num_users        = params.num_users
        self.num_items        = params.num_items    
        self.reg_lambda       = params.reg_lambda
        self.reg_Wh           = params.reg_Wh
        self.rating_scale     = params.rating_scale
        self.initializer      = params.initializer
        
    def define_model(self,user_indices,user_cdcf_indices,item_indices,
                                   true_rating,keep_prob,keep_prob_layer,domain_indices):
        self.user_indices       = user_indices
        self.user_cdcf_indices  = user_cdcf_indices
        self.item_indices       = item_indices
        self.true_rating        = true_rating
        self.keep_prob          = keep_prob
        self.keep_prob_layer    = keep_prob_layer
        self.domain_indices     = domain_indices
        
        # variables
        self.user_embeddings      = tf.Variable(self.initializer(shape=[self.num_users,
                                    self.num_factors]),
                                    dtype=tf.float32,name='user_embedding')
        # cdcf initialize with zero
        self.user_cdcf_embeddings = tf.Variable(self.initializer(shape=[2*self.num_users, 
                                    self.num_factors]),
                                    dtype=tf.float32,name='user_cdcf_embedding')
        self.item_embeddings      = tf.Variable(self.initializer(shape=[self.num_items, 
                                    self.num_factors]),
                                    dtype=tf.float32,name='item_embedding')
        self.W_G = tf.Variable(self.initializer(shape=[self.num_factors,1]),
                               dtype=tf.float32,name='W_G')
        self.W_G_dp       = tf.nn.dropout(self.W_G, self.keep_prob)
        self.bias_G = tf.Variable(tf.zeros(shape=[1,1]),
                               dtype=tf.float32,name='bias_G')
        
        # definitions
        self.user_embeds      = tf.nn.embedding_lookup(self.user_embeddings, self.user_indices)
        self.user_embeds      = tf.nn.dropout (self.user_embeds, self.keep_prob_layer)
        self.user_cdcf_embeds = tf.nn.embedding_lookup(self.user_cdcf_embeddings, 
                                                       self.user_cdcf_indices)
        self.user_cdcf_embeds = tf.nn.dropout (self.user_cdcf_embeds, self.keep_prob_layer)
        self.item_embeds      = tf.nn.embedding_lookup(self.item_embeddings, self.item_indices)
        self.item_embeds      = tf.nn.dropout (self.item_embeds, self.keep_prob_layer)
        
        self.user_add_output   = tf.add(self.user_embeds,self.user_cdcf_embeds) # cat 

        self.multiplied_output = tf.multiply(self.user_add_output,self.item_embeds)
        self.multiplied_output = tf.nn.dropout(self.multiplied_output, self.keep_prob_layer)
        
        self.temp_pred_rating  = (self.rating_scale * tf.nn.sigmoid 
                                  (tf.matmul(self.multiplied_output,self.W_G_dp)))
        self.pred_rating       = tf.reshape(self.temp_pred_rating,shape=[-1])
                                      
    def define_loss(self,loss_type='all'):
        self.regularization_loss = (self.reg_lambda * (tf.nn.l2_loss(self.user_embeds) +
                                                 tf.nn.l2_loss(self.item_embeds) +
                                                 tf.nn.l2_loss(self.user_cdcf_embeds))+
                                    self.reg_Wh *  (tf.nn.l2_loss(self.W_G)))     
