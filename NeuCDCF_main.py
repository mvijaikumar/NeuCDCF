import tensorflow as tf
import numpy as np
import sys
import math
import argparse
from time import time
sys.path.append('./.')
np.random.seed(7)
tf.set_random_seed(7)
from Dataset import Dataset
from Dataset_cold_start import Dataset_cold_start
from Error_plot import Error_plot
from Utilities import is_stopping_criterion_reached
from Batch import Batch
from Parameters import Parameters
from NeuCDCF import NeuCDCF
from Arguments import parse_args
from Pretrain import Pretrain

if __name__ == '__main__':
    args = parse_args()
    print(args)
    filepath      = args.path + args.dataset
    result_file   = args.res_file
    cdcf_flag     = True
    
    # Data processing
    t1 = time()
    print('Data loading...')
    dataset = Dataset(filepath,cdcf_flag)
        
    user_input,item_input,train_rating,train_domain               = dataset.trainArrQuadruplets
    valid_user_input, valid_item_input, valid_rating,valid_domain = dataset.validArrQuadruplets
    test_user_input, test_item_input, test_rating,test_domain     = dataset.testArrQuadruplets
    params = Parameters(args,dataset)

    #cdcf
    user_cdcf_input       = params.num_users * train_domain + user_input
    valid_user_cdcf_input = params.num_users * valid_domain + valid_user_input
    test_user_cdcf_input  = params.num_users * test_domain  + test_user_input
    
    params.set_input(user_input,item_input,train_rating,train_domain,user_cdcf_input,
        valid_user_input, valid_item_input, valid_rating,valid_domain,valid_user_cdcf_input,
        test_user_input, test_item_input, test_rating,test_domain,test_user_cdcf_input)
    
    print("""Load data done [%.1f s]. #user:%d, #item:%d, #src_item:%d,#tar_item:%d,
          #train:%d, #test:%d, #valid:%d"""
          % (time() - t1, params.num_users,params.num_items,params.num_source_items,
             params.num_target_items,params.num_train_instances,
             params.num_test_instances,params.num_valid_instances))
    
    print("Method: %s"%(params.method.upper()))    
    t1 = time()
    
    model = NeuCDCF(params)
    model.define_model()
    model.define_loss('all')
    print "Model definition completed: ",time()-t1
    error_plot = Error_plot(type='error') #error or accuracy
    
    # optimization
    train_step = tf.train.RMSPropOptimizer(params.learning_rate).minimize(model.loss)
    init = tf.global_variables_initializer()  
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        pretrain = Pretrain(params,sess)
        sess.run(init)
        if params.pretrain_load == True:
            pretrain.load(model)

        best_mae,   best_test_mae,best_iter   = 100,100, -1
        best_mse, best_test_mse,best_mse_iter = 200,200, -1
        history_mae = []

        batch = Batch(params.num_train_instances,params.batch_size,shuffle=True)
        batch_valid = Batch(params.num_valid_instances,params.batch_size,shuffle=False)
        batch_test  = Batch(params.num_test_instances,params.batch_size,shuffle=False)

        for epoch_num in range(params.num_epochs+1):
            t1 = time()
            mae_train,mse_train = 0.0,0.0
            loss,mse_loss,reg_loss,recon_loss = 0.0,0.0,0.0,0.0
            while batch.has_next_batch():
                shuff_batch = batch.get_next_batch_indices()
                bsiz = len(shuff_batch)

                feed_dict_train = {model.user_indices:user_input[shuff_batch],
                           model.user_cdcf_indices:user_cdcf_input[shuff_batch],
                           model.item_indices:item_input[shuff_batch],
                           model.true_rating:train_rating[shuff_batch],
                           model.domain_indices:train_domain[shuff_batch],
                           model.keep_prob:params.dp_keep_prob,
                                   model.keep_prob_layer:params.keep_prob_layer,
                                   model.valid_clip:0.0}
            
                (_,batch_loss,batch_reg_err,batch_recon_err,
                 batch_mse_train,batch_mae_train) = sess.run([train_step,model.loss,
                                        model.regularization_loss,model.recon_error,
                                        model.mse_loss,model.mae_loss,
                                        ], feed_dict=feed_dict_train)
                mae_train += batch_mae_train * bsiz
                mse_train += batch_mse_train * bsiz
                
                loss       += batch_loss
                reg_loss   += batch_reg_err
                recon_loss += batch_recon_err
                mse_loss   += batch_mse_train
                
            mae_train = mae_train/params.num_train_instances
            mse_train = mse_train/params.num_train_instances
            batch.initialize_next_epoch()
            
            if epoch_num%1==0:
                mae_valid,mse_valid = 0.0,0.0
                mae_test,mse_test = 0.0,0.0
                while batch_valid.has_next_batch():
                    valid_batch_indices = batch_valid.get_next_batch_indices()
                    bsiz = len(valid_batch_indices)
                    feed_dict_valid = {model.user_indices:valid_user_input[valid_batch_indices],
                           model.user_cdcf_indices:valid_user_cdcf_input[valid_batch_indices],
                           model.item_indices:valid_item_input[valid_batch_indices],
                           model.true_rating:valid_rating[valid_batch_indices],
                           model.domain_indices:valid_domain[valid_batch_indices],
                           model.keep_prob:1.0,
                                   model.keep_prob_layer:1.0,
                                       model.valid_clip:1.0}
                    
                    batch_mae_valid,batch_mse_valid = sess.run([model.mae_loss,model.mse_loss],
                                               feed_dict=feed_dict_valid)##
                    mae_valid += batch_mae_valid * bsiz
                    mse_valid += batch_mse_valid * bsiz 
                mae_valid = mae_valid/params.num_valid_instances
                mse_valid = mse_valid/params.num_valid_instances
                
                while batch_test.has_next_batch():
                    test_batch_indices = batch_test.get_next_batch_indices()
                    bsiz = len(test_batch_indices)
                    feed_dict_test = {model.user_indices:test_user_input[test_batch_indices],
                           model.user_cdcf_indices:test_user_cdcf_input[test_batch_indices],
                           model.item_indices:test_item_input[test_batch_indices],
                           model.true_rating:test_rating[test_batch_indices],
                           model.domain_indices:test_domain[test_batch_indices],
                           model.keep_prob:1.0,
                                   model.keep_prob_layer:1.0,
                                      model.valid_clip:1.0}
                    
                    batch_mae_test,batch_mse_test = sess.run([model.mae_loss,model.mse_loss],
                                               feed_dict=feed_dict_test)
                    mae_test += batch_mae_test * bsiz
                    mse_test += batch_mse_test * bsiz
                mae_test = mae_test/params.num_test_instances
                mse_test = mse_test/params.num_test_instances
                
                batch_valid.initialize_next_epoch()
                batch_test.initialize_next_epoch()
                
                print("""[%.2f s] iter:%3i train loss:%.4f train mse loss:%.4f train reg loss:%.4f recon loss:%.4f """
                        %(time()-t1,epoch_num,loss,mse_loss,reg_loss,recon_loss))

                rmse_train,rmse_valid,rmse_test = math.sqrt(mse_train),math.sqrt(mse_valid),math.sqrt(mse_test)
                print("""mae_train: %.4f rmse_train: %.4f mae_valid : %.4f rmse_valid : %.4f mae_tst : %.4f rmse_tst : %.4f"""
                        %(mae_train,math.sqrt(mse_train),mae_valid,math.sqrt(mse_valid),
                          mae_test,math.sqrt(mse_test)))

                error_plot.append(loss,recon_loss,reg_loss,mse_loss,mae_train,mae_valid,mae_test,
                                rmse_train,rmse_valid,rmse_test)

                history_mae.append(mae_valid)
            if mae_valid < best_mae:
                best_mae, best_iter,best_test_mae = mae_valid, epoch_num, mae_test
                if params.pretrain_save == True:
                    pretrain.save(model)

            if mse_valid < best_mse:
                best_mse, best_mse_iter,best_test_mse = mse_valid, epoch_num, mse_test
            
            if params.stopping_criteria == True:
                if is_stopping_criterion_reached(history_mae):
                    print 'Stopping criterion reached.'
                    break
                    
            
        plot_path =  result_file.replace(result_file.split('/')[-1],"") + params.method + '/' + args.path.replace("//","/").replace("/","_") + 'fact'+str(params.num_factors)
        print "plot path: " + plot_path
        error_plot.plot(True,plot_path)
        
        print("""End. Best Iteration mae %d: val mae = %.4f  tst mae = %.4f mse_itr %d:  
        val rmse = %.4f  test rmse = %.4f""" % (best_iter, best_mae,best_test_mae ,best_mse_iter,
                 math.sqrt(best_mse), math.sqrt(best_test_mse)))
        result_fp = open(result_file,'a')
        result_fp.write("""bst_iter %d: valid_mae = %.4f test_mae = %.4f mse_itr %d: valid_rmse = %.4f test_rmse = %.4f %s \n"""% (best_iter, best_mae,best_test_mae,
                best_mse_iter,math.sqrt(best_mse),math.sqrt(best_test_mse),args))
        result_fp.close()
