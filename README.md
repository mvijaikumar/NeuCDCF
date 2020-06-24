# NeuCDCF
Source code for the paper **Neural Cross Domain Collaborative Filtering with Shared Entities (NeuCDCF)**

Author: Vijaikumar M *et.al.*

## Requirements
1. python 2.7.12
2. scipy 1.0.0
3. numpy 1.13.3
4. tensorflow 1.12.0
## Example to run the codes
### GCMF
python NeuCDCF_main.py --method gcmf
### SED
python NeuCDCF_main.py --method sed
### NeuCDCF
python NeuCDCF_main.py --method neucdcf
### To know about more options
python NeuCDCF_main.py --help

Look at the Arguments.py file for an example configuration
## Datasets

Format: user_id item_id rating domain_index

Amazon Movie (source)-Book (target)  link: https://www.dropbox.com/sh/sg7kknun6bm0vm9/AACC6JhIiW1lDIFWl9BknDAga?dl=0 

Amazon Movie (source)-Music (target) link: https://www.dropbox.com/sh/dhwu8jxpwlmt7rt/AACDdbVBj185QmDOyE02ctJXa?dl=0


Douban Movie (soure)-Book (target) link: will be provided on sending e-mail to vijaikumar@iisc.ac.in

Douban Music (soure)-Book (target) link: will be provided on sending e-mail to vijaikumar@iisc.ac.in
