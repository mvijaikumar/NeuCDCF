import numpy as np
import matplotlib.pyplot as plt
import distutils.dir_util
import subprocess

class Error_plot(object):
    def __init__(self,type='error'):
        if type=='error':
            self.loss_list,self.recon_loss_list,self.reg_loss_list,self.mse_loss_list = [],[],[],[]
            self.train_mae_list,self.valid_mae_list,self.test_mae_list             = [],[],[]
            self.train_rmse_list,self.valid_rmse_list,self.test_rmse_list          = [],[],[]
        
    def append(self,loss,recon_loss,reg_loss,mse_loss,train_mae,valid_mae,test_mae,train_rmse,valid_rmse,test_rmse):
        self.loss_list.append(loss)
        self.reg_loss_list.append(reg_loss)
        self.recon_loss_list.append(recon_loss)
        self.mse_loss_list.append(mse_loss)
        
        self.train_mae_list.append(train_mae)
        self.valid_mae_list.append(valid_mae)
        self.test_mae_list.append(test_mae)
        self.train_rmse_list.append(train_rmse)
        self.valid_rmse_list.append(valid_rmse)
        self.test_rmse_list.append(test_rmse)
        
    def plot(self,save_flag=False,path=None):
        iter_count = len(self.loss_list)
        print 'iter_count: ',iter_count
        self.iterations = range(iter_count)
        if save_flag:
            subprocess.call(['mkdir','-p',path.replace(path.split('/')[-1],"")])
        
        self.plot_train_error(save_flag,path)
        self.plot_rmse_error(save_flag,path)
        self.plot_mae_error(save_flag,path)
        
    def plot_train_error(self,save_flag, path):
        plt.clf()
        plt.plot(self.iterations, self.loss_list ,'r--',label='total loss')
        plt.plot(self.iterations, self.recon_loss_list, 'b--',label='recon loss')
        plt.plot(self.iterations, self.reg_loss_list, 'g--',label='reg loss')
        plt.plot(self.iterations,self.mse_loss_list,'c--',label='mse loss')
        
        plt.xlabel('Iteration')
        plt.ylabel('Error values')
        plt.title('Iteration vs training error')
        plt.legend()
        plt.grid(True)
        if save_flag:
            plt.savefig(path+'train.png', bbox_inches='tight')
        else:
            plt.show()
        
    def plot_rmse_error(self,save_flag,path):
        plt.clf()
        plt.plot(self.iterations, self.train_rmse_list,'r-', label='train rmse')
        plt.plot(self.iterations, self.valid_rmse_list, 'b-',label='valid rmse')
        plt.plot(self.iterations, self.test_rmse_list, 'g-', label='test rmse')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE values')
        plt.title('Iteration vs RMSE  error')
        plt.legend()
        plt.grid(True)
        if save_flag:
            plt.savefig(path+'rmse.png', bbox_inches='tight')
        else:
            plt.show()
        
    def plot_mae_error(self,save_flag,path):
        plt.clf()
        plt.plot(self.iterations, self.train_mae_list ,'r-',label='train mae')
        plt.plot(self.iterations, self.valid_mae_list, 'b-',label='valid mae')
        plt.plot(self.iterations, self.test_mae_list, 'g-',label='test mae')
        
        plt.xlabel('Iteration')
        plt.ylabel('MAE values')
        plt.title('Iteration vs MAE error')
        plt.legend()
        plt.grid(True)
        if save_flag:
            plt.savefig(path+'mae.png', bbox_inches='tight')
        else:
            plt.show()
