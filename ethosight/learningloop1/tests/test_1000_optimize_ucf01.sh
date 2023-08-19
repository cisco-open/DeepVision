# runs a test on the small set of ucf images that were manually extracted by shan
EthosightAppCLI optimize testapp_ucf01 |tee test_1000_output.txt 

# with affinity_minthreshold = 26
# no lso
#Top 1 accuracy: 64.29%
#Top 5 accuracy: 78.57%
# with lso
#Top 1 accuracy: 7.14%
#Top 5 accuracy: 35.71%
