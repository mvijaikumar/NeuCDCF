import numpy as np
import tensorflow as tf
class Parameters(object):
    def __init__(self,args,dataset):
        self.method           = args.method
        self.num_factors      = args.num_factors
        self.num_epochs       = args.epochs
        self.learning_rate    = args.lr
        self.reg_lambda       = args.reg_lambda
        self.batch_size       = args.batch_size
        self.reg_Wh           = args.reg_Wh
        self.reg_bias         = args.reg_bias
        self.rating_scale     = args.rating_scale
        self.dp_keep_prob     = args.dp_keep_prob
        self.alpha            = args.alpha
        
        self.num_users        = dataset.number_of_users
        self.num_items        = dataset.number_of_items
        self.num_source_items = dataset.number_of_source_items
        self.num_target_items = dataset.number_of_target_items
        #self.source_shape     = (dataset.number_of_users,dataset.number_of_source_items)
        #self.target_shape     = (dataset.number_of_users,dataset.number_of_target_items)
        
        self.num_train_instances = len(dataset.trainArrQuadruplets[0])
        self.num_valid_instances = len(dataset.validArrQuadruplets[0])
        self.num_test_instances  = len(dataset.testArrQuadruplets[0])
        
        self.method     = args.method
        if args.cdcf=='yes': 
            self.cdcf_flag  = True
        else:
            self.cdcf_flag = False
            
        if args.stopping_criteria == 1:
            self.stopping_criteria = True
        else:
            self.stopping_criteria = False
            
            
        self.train_source = dataset.trainMatrix_source.tocoo()
        self.train_target = dataset.trainMatrix_target.tocoo()
    
        self.source_indices = np.array([self.train_source.row,self.train_source.col]).T
        self.source_values  = np.array(self.train_source.data)
        self.source_shape   = (self.num_users,self.num_source_items)
    
        self.target_indices = np.array([self.train_target.row,self.train_target.col]).T
        self.target_values  = np.array(self.train_target.data)
        self.target_shape   = (self.num_users,self.num_target_items)
        
        # new 
        self.mu        = np.mean(dataset.trainArrQuadruplets[2])
        self.mu_source = np.mean(self.source_values)
        self.mu_target = np.mean(self.target_values)
        
        # initializations
        if args.initializer == 'xavier':
            print('Initializer: xavier')
            self.initializer = tf.contrib.layers.xavier_initializer()
        elif args.initializer == 'random_normal':
            print('Initializer: random_normal')
            _stddev = args.stddev
            self.initializer = tf.random_normal_initializer(stddev=_stddev)
        elif args.initializer == 'random_uniform':
            print('Initializer: random_uniform')
            _min,_max = -args.stddev, args.stddev
            self.initializer = tf.random_uniform_initializer(minval=_min,maxval=_max)
        
        if args.pretrain_save == 1:
            self.pretrain_save = True
        else:
            self.pretrain_save = False
        if args.pretrain_load == 1:
            self.pretrain_load = True
        else:
            self.pretrain_load = False
        self.pretrain_path = args.pretrain_path
        #self.valid_clip = args.clip
        self.rating_min_val   = args.rating_min_val
        self.rating_max_val   = args.rating_max_val
        if args.analysis == 1:
            self.analysis   = True
        else:
            self.analysis   = False
        
        self.keep_prob_layer = args.keep_prob_layer
        
    def set_input(self,user_input,item_input,train_rating,train_domain,user_cdcf_input,
              valid_user_input, valid_item_input, valid_rating,valid_domain,valid_user_cdcf_input,
              test_user_input, test_item_input, test_rating,test_domain,test_user_cdcf_input):
        
        self.user_input = user_input
        self.item_input = item_input
        self.train_rating = train_rating
        self.train_domain = train_domain
        self.user_cdcf_input = user_cdcf_input
        self.valid_user_input = valid_user_input
        self.valid_item_input = valid_item_input 
        self.valid_rating = valid_rating
        self.valid_domain = valid_domain
        self.valid_user_cdcf_input = valid_user_cdcf_input
        self.test_user_input = test_user_input
        self.test_item_input = test_item_input 
        self.test_rating = test_rating
        self.test_domain = test_domain
        self.test_user_cdcf_input = test_user_cdcf_input
        