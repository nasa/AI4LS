#!/bin/bash

#SESSION=fed
N_COLABS=${1:-"5"}
DETACH_SESSION=${2:-false}
DIR=${3:-"/home/user/data"}
SESSION=${4:-"fed"}

WORKSPACE=fl_workspace

tmux new-session -d -s $SESSION
tmux rename-window -t 0 $SESSION

tmux select-window -t $SESSION:0
    tmux split-window -v

counter=0
while [ $counter -lt ${N_COLABS} ]
do
    echo $counter

    tmux split-window -h
    tmux select-pane -t 1

    ((counter++))
done

tmux select-pane -t 0
#tmux send-keys "conda activate xxxx" C-m
tmux send-keys "mkdir $DIR/save" C-m
tmux send-keys "mkdir $DIR/save/log" C-m

tmux send-keys "cd $WORKSPACE" C-m
tmux send-keys "fx -f save/logs.txt aggregator start" C-m
tmux send-keys "fx model save -m save/crisp_last_.pbuf" C-m
tmux send-keys "cp save/* $DIR/save/" C-m
tmux send-keys "cp plan/* $DIR/plan/" C-m
if [ "$DETACH_SESSION" == true ]; then
    tmux send-keys "tmux detach -s $SESSION" C-m
fi



counter=0
pane=1
while [ $counter -lt ${N_COLABS} ]
do
    echo $counter
    tmux select-pane -t $pane
    #tmux send-keys "conda activate xxxx" C-m
    tmux send-keys "cd $WORKSPACE/col_$counter/$WORKSPACE" C-m
    tmux send-keys "fx -f save/col_$counter.txt collaborator start -n col_$counter" C-m
    tmux send-keys "cp save/col_$counter.txt $DIR/save/log/" C-m

    ((counter++))
    ((pane++))
done

tmux select-pane -t 0

# Attach to session
tmux attach-session -t $SESSION:0
