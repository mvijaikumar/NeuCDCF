def is_stopping_criterion_reached(mae_list):
    ''' function to stop the iterations when stopping 
    criterian is reached'''
    num_iter = len(mae_list)
    if num_iter<6:
        return False
    elif np.min(mae_list[-6:-1]) < np.min(mae_list[-5:]):
        return True
    return False