import argparse
import sys
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuCDCF.")    
    parser.add_argument('--method', nargs='?', default='gcmf', help='gcmf,sed,neucdcf')
    parser.add_argument('--path', nargs='?',
               #default='./data/amazon/cd_movie/sparse/100/fold1/',
               default='data/amazon/music_movie/sparse/100/fold1/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='cd',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=120,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--reg_Wh', type=float, default=0.0000,
                       help="Regularization for weight vector.")
    parser.add_argument('--reg_bias', type=float, default=0.000,
                       help="Regularization for user and item bias embeddings.")
    parser.add_argument('--reg_lambda', type=float, default=0.000, #5
                       help="Regularization lambda for user and item embeddings.")
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='rmsprop',
                        help='Specify an optimizer: rmsprop')
    parser.add_argument('--initializer', nargs='?', default='random_normal',
                        help='random_normal,random_uniform,xavier')
    parser.add_argument('--verbose', type=int, default=1, 
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--stddev', type=float, default=0.002,
                        help='stddev for normal and [min,max] for uniform')
    parser.add_argument('--rating_scale', type=float, default=5.0,
                        help='Rating scale for example max = 5 [min,max].')
    parser.add_argument('--rating_min_val', type=float, default=1.0,
                        help='Rating scale for example min = 1.0 [min,max].')
    parser.add_argument('--rating_max_val', type=float, default=5.0,
                        help='Rating scale for example max = 5.0 [min,max].')
    parser.add_argument('--res_file', nargs='?', default='./gcmf.res',
                        help='result file path.')
    parser.add_argument('--dp_keep_prob', type=float, default=1.0,
                        help='droupout keep probability.') #9
    parser.add_argument('--keep_prob_layer', type=float, default=0.5,
                        help='droupout keep probability in layers.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='for gmf vs reconstruction.')
    parser.add_argument('--stopping-criteria', type=int, default=0,
                        help='(yes-1,no-0).')
    parser.add_argument('--pretrain_save', type=int, default=0,
                        help='(yes-1,no-0).')
    parser.add_argument('--pretrain_load', type=int, default=0,
                        help='(yes-1,no-0).')
    parser.add_argument('--pretrain_path', nargs='?', default='./pretrain',
                        help='path to pretrain directory.')
    return parser.parse_args()
