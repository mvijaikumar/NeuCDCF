import tensorflow as tf
import numpy as np

class SED(object):
    def __init__(self,params):
        self.num_factors      = params.num_factors        
        self.num_users        = params.num_users
        self.num_items        = params.num_items
        self.num_source_items = params.num_source_items
        self.num_target_items = params.num_target_items
        self.source_shape     = params.source_shape
        self.target_shape     = params.target_shape
        self.reg_lambda       = params.reg_lambda
        self.reg_Wh           = params.reg_Wh
        self.rating_scale     = params.rating_scale
        
        self.source_indices   = params.source_indices
        self.source_values    = params.source_values
        self.target_indices   = params.target_indices
        self.target_values    = params.target_values
        self.initializer      = params.initializer
        self.noise_stddev     = params.noise

            
    def define_model(self,user_indices,user_cdcf_indices,item_indices,
                                   true_rating,keep_prob,keep_prob_layer,domain_indices,
                    cur_batch_size,valid_clip):
        
        self.user_indices       = user_indices
        self.user_cdcf_indices  = user_cdcf_indices
        self.item_indices       = item_indices
        self.true_rating        = true_rating
        self.keep_prob          = keep_prob
        self.keep_prob_layer    = keep_prob_layer
        self.domain_indices     = domain_indices
        self.cur_batch_size     = cur_batch_size #new
        self.valid_clip         = valid_clip #new
        
        self.b_u1 = tf.Variable(tf.zeros(shape=[1,1]),dtype=tf.float32,name='b_u1')
        self.b_u2 = tf.Variable(tf.zeros(shape=[1,1]),dtype=tf.float32,name='b_u2')
        self.b_v1 = tf.Variable(tf.zeros(shape=[1,1]),dtype=tf.float32,name='b_v1')
        
        
        self.user_source_embeddings = tf.Variable(tf.sparse_to_dense(self.source_indices,
                                    self.source_shape,self.source_values,0,True,name='spd1'), 
                                    trainable=False,dtype=tf.float16,name='user_source_embedding')
        self.user_target_embeddings = tf.Variable(tf.sparse_to_dense(self.target_indices,
                                    self.target_shape,self.target_values,0,True,name='spd2'), 
                                    trainable=False,dtype=tf.float16,name='user_target_embedding')
        
        self.W_AE = tf.Variable(self.initializer(shape=[2*self.num_factors,1]),
                               dtype=tf.float32,name='W_AE')
        
        self.item_embeddings      = tf.Variable(self.initializer(shape=[self.num_items, 
                                     4*self.num_factors]),dtype=tf.float32,name='item_embedding')
        
        self.item_embeds        = tf.nn.embedding_lookup(self.item_embeddings,self.item_indices)  
        self.item_embeds_dp     = tf.nn.dropout(self.item_embeds, self.keep_prob_layer)
        self.user_source_embeds = tf.cast(tf.nn.embedding_lookup(self.user_source_embeddings, 
                                        self.user_indices),tf.float32)
        self.user_target_embeds = tf.cast(tf.nn.embedding_lookup(self.user_target_embeddings, 
                                        self.user_indices),tf.float32)
        self.user_target_mask   = tf.greater(self.user_target_embeds,0)
        # new =============================
        #self.user_source_mask   = tf.greater(self.user_source_embeds,0)
        #self.noise              = tf.random_normal(mean=0.0,stddev=self.noise_stddev,
        #                                shape=[self.cur_batch_size,self.num_source_items])
        #self.noise_masked       = tf.multiply(tf.cast(self.user_source_mask,dtype=tf.float32),self.noise)
        #self.noise#tf.boolean_mask(self.noise,self.user_source_mask)
        #self.user_source_embeds = (self.user_source_embeds + (1-self.valid_clip) * self.noise_masked)
        # ===================================
        
        self.Weight_v1    = tf.Variable(self.initializer(shape=[4*self.num_factors,
                            2*self.num_factors]),dtype=tf.float32,name='Weight_v1')
        self.Weight_u1    = tf.Variable(self.initializer(shape=[self.num_source_items,
                            2*self.num_factors]),dtype=tf.float32,name='Weight_u1')
        self.Weight_u2    = tf.Variable(self.initializer(shape=[2*self.num_factors,
                           self.num_target_items]),dtype=tf.float32,name='Weight_u2')        
        
        self.item_rep1    = tf.nn.sigmoid(tf.matmul(self.item_embeds_dp,
                                                    self.Weight_v1) + self.b_v1)
        self.item_rep1_dp = tf.nn.dropout(self.item_rep1, self.keep_prob_layer)        
        
        self.user_rep1    = tf.nn.sigmoid(tf.matmul(self.user_source_embeds,
                                                    self.Weight_u1) + self.b_u1)
        self.user_rep1_dp = tf.nn.dropout(self.user_rep1, self.keep_prob_layer)        
        self.user_rep2    = tf.nn.sigmoid(tf.matmul(self.user_rep1_dp,
                                                   self.Weight_u2) + self.b_u2) # L/2 representation
        self.user_rep2_dp = tf.nn.dropout(self.user_rep2, self.keep_prob_layer)        
        
        self.user_target_true  = tf.boolean_mask(self.user_target_embeds,self.user_target_mask)
        self.user_target_pred  = tf.boolean_mask(self.user_rep2,self.user_target_mask)

        self.multiplied_output = tf.multiply(self.user_rep1_dp,self.item_rep1_dp) 
        #self.multiplied_output = tf.nn.dropout(self.multiplied_output,self.keep_prob_layer)
        
        self.temp_pred_rating  = self.rating_scale * tf.nn.sigmoid(tf.matmul(self.multiplied_output,
                                                                       self.W_AE)) 
        self.pred_rating       = tf.reshape(self.temp_pred_rating,shape=[-1])            
        
    def define_loss(self,loss_type='all'):
        # add item weight vectors
        val = 1
        print "hp infront of recon:",val
        self.recon_error         = val * tf.reduce_mean(tf.pow(tf.subtract(self.user_target_true,
                                                                  self.user_target_pred),2))
        self.regularization_loss = (self.reg_lambda * (tf.nn.l2_loss(self.item_embeds))+
                                    self.reg_Wh *  (tf.nn.l2_loss(self.Weight_u1)+
                                                    tf.nn.l2_loss(self.Weight_u2)+
                                                    tf.nn.l2_loss(self.Weight_v1)+
                                                    tf.nn.l2_loss(self.W_AE)))
