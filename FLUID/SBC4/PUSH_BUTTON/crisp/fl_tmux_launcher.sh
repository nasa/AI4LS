#!/bin/bash

N_COLABS=${1:-"5"}
DETACH_SESSION=${2:-false}
DIR=${3:-"/home/user/data"}
SESSION=${4:-"fed"}

WORKSPACE=fl_workspace
tmux new-session -d -s $SESSION

window=0
tmux select-window -t $SESSION:$window
tmux select-pane -t 0
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

col=0
while [ $col -lt ${N_COLABS} ]
do
((window++))
tmux new-window -t $SESSION:$window
tmux select-window -t $SESSION:$window

	counter=0
	while [ $counter -lt 5 ]
	do
	    tmux split-window -v
	    tmux select-pane -t 0
	    ((counter++))
	done

	counter=0
	while [ $counter -lt 5 ]
	do
        tmux select-pane -t $counter
	    tmux send-keys "cd $WORKSPACE/col_$col/$WORKSPACE" C-m
	    tmux send-keys "fx -f save/col_$col.txt collaborator start -n col_$col" C-m
	    tmux send-keys "cp save/col_$col.txt $DIR/save/log/" C-m

	    ((counter++))
	    ((col++))
	done
done

tmux select-pane -t 0

# Attach to session
tmux attach-session -t $SESSION:0