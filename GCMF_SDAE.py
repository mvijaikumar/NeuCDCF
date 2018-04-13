class GCMF_SDAE(object):
    def __init__(self,params):
        self.params           = params 
        self.alpha            = params.alpha
        self.rating_scale     = params.rating_scale 

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
        
        self.gcmf_model  = GCMF(self.params)
        self.gcmf_model.define_model(self.user_indices,self.user_cdcf_indices,
                                        self.item_indices,self.true_rating,
                                        self.keep_prob,self.keep_prob_layer,self.domain_indices)
        
        self.sdae_model  = SDAE(self.params)
        self.sdae_model.define_model(self.user_indices,self.user_cdcf_indices,
                                        self.item_indices,self.true_rating,
                                        self.keep_prob,self.keep_prob_layer,self.domain_indices,
                     self.cur_batch_size,self.valid_clip)
        
        self.gcmf_rep      = self.gcmf_model.multiplied_output
        self.sdae_rep      = self.sdae_model.multiplied_output
        self.gcmf_sdae_rep = tf.concat([self.gcmf_rep,self.sdae_rep], axis=1)
        
        self.W_GAE = tf.concat([(1-self.alpha) * self.gcmf_model.W_G,
                                self.alpha * self.sdae_model.W_AE], axis=0)
        
        self.temp_pred_rating  = self.rating_scale * tf.nn.sigmoid(
                                 tf.matmul(self.gcmf_sdae_rep,self.W_GAE))
        self.pred_rating       = tf.reshape(self.temp_pred_rating,shape=[-1])
        
                                      
    def define_loss(self,loss_type='all'):
        self.gcmf_model.define_loss(loss_type=loss_type)
        self.sdae_model.define_loss(loss_type=loss_type)
        self.regularization_loss = (self.gcmf_model.regularization_loss + 
                                    self.sdae_model.regularization_loss)   
        self.recon_error = self.sdae_model.recon_error
