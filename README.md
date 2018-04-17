# NeuCDCF
Source code for Neural Cross Domain Collaborative Filtering with Shared Resources (NeuCDCF)
## Requirements
1. python 2.7.12
2. scipy 1.0.0
3. numpy 1.13.3
4. tensorflow 1.4.0
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
## Dataset format (Tab separated)
userid  itemid  rating  domain_index

Amazon Movie (source)-Book (target)  link: will be updated soon 

Amazon Movie (source)-Music (target) link: will be updated soon


Douban Movie (soure)-Book (target) link: will be updated soon

Douban Music (soure)-Book (target) link: will be updated soon
