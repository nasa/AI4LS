PROTOBUF_FILE=../data/WORKSPACE/AGG_EARTH/workspace/workspace/save/crisp_best_nlerm.pt
TRAINDATA_FILE=../data/WORKSPACE/COLAB_EARTH/col_0/train/data.csv



#python get_results.py -pf $PROTOBUF_FILE -tf $TRAINDATA_FILE -nf 10 -mn ERM -pw False
python get_results.py -pf $PROTOBUF_FILE -tf $TRAINDATA_FILE -mn ERM -pw False
