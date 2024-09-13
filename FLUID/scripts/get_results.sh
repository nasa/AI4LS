PROTOBUF_FILE=/home/fluid/data/WORKSPACE/workspace/save/crisp_best_nlerm.pt
TRAINDATA_FILE=/home/fluid/data/col_0/train/data.csv



python get_results.py -pf $PROTOBUF_FILE -tf $TRAINDATA_FILE -nf 10 -mn ERM -pw False
