import numpy as np
import scipy.sparse as sp

# rating files are quadruplets and of 0 indexed with no gap between ids
class Dataset(object):
    def __init__(self, path, cdcf_flag=False):
        train_suffix = ".train.rating"
        print 'path: ' + path
        path   = path.replace("//","/")
        train_suffix = "_"+path.split("/")[-5].split("_")[1]+ train_suffix
        print ('dataset path: ' + path + train_suffix)
        self.get_user_item_count(path + train_suffix)
        self.get_item_count_domainwise(path + train_suffix)
        self.trainMatrix_target,self.trainMatrix_source = (self.load_rating_file_as_matrix_train(path 
                                                            + train_suffix))        
        self.trainArrQuadruplets = self.load_rating_file_as_arraylist(path + train_suffix)
        self.validArrQuadruplets = self.load_rating_file_as_arraylist(path + ".valid.rating")
        self.testArrQuadruplets  = self.load_rating_file_as_arraylist(path + ".test.rating")                       
    
    def get_user_item_count(self, filename):
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        self.number_of_users = num_users+1
        self.number_of_items = num_items+1
        
    def get_item_count_domainwise(self, filename):
        num_items = 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                i, dom = int(arr[1]), int(arr[3])
                if dom == 0:
                    #num_users = max(num_users, u)
                    num_items = max(num_items, i)
                line = f.readline()
        self.number_of_target_items = num_items+1
        self.number_of_source_items = self.number_of_items - self.number_of_target_items

    def load_rating_file_as_matrix(self, filename):
        # Construct matrix
        mat = sp.lil_matrix((self.number_of_users, self.number_of_target_items), dtype=np.float16)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = rating
                line = f.readline()    
        return mat

    def load_rating_file_as_matrix_train(self, filename):
        # Construct matrix
        mat1 = sp.lil_matrix((self.number_of_users, self.number_of_target_items), dtype=np.float16)
        mat2 = sp.lil_matrix((self.number_of_users, self.number_of_source_items), dtype=np.float16)
        
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating, dom = int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3])
                if (rating > 0 and dom==0):
                    mat1[user, item] = rating
                elif(rating > 0 and dom ==1):
                    mat2[user, item - self.number_of_target_items] = rating
                line = f.readline()    
        return mat1,mat2
    
    def load_rating_file_as_arraylist(self, filename):
        user_input, item_input, rating, domain = [],[],[],[]
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rat, dom_num = int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3])
                user_input.append(user)
                item_input.append(item)
                rating.append(rat)
                domain.append(dom_num)
                line = f.readline()
        return np.array(user_input), np.array(item_input), np.array(rating,dtype=np.float16), np.array(domain)
