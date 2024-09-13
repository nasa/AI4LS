#!/bin/bash

# Test harness script for the comms-updown.sh script, which simulates the arrival of data files on the bridge node,
# specifically, the TO_BRIDGE_FROM_COLABSHIM and TO_BRIDGE_FROM_TDS directories.
# File generation defaults to 1 file every 10 seconds, but can be altered with a numeric parameter (e.g. fluid_commstest 20 = every 20 seconds)
# File generation continues until the user presses any key.

FLUID_HOME=/home/fluid     # Use this to require that this script ONLY run under the "fluid" user account
# FLUID_HOME=~    # Use this to test this script under any user account (e.g. ec2-user)

TO_BRIDGE_FROM_COLABSHIM=$FLUID_HOME/TO_BRIDGE_FROM_COLABSHIM   # this will be a symbolic link and used to simulate comms failure for data being downlinked from the ISS
TO_BRIDGE_FROM_TDS=$FLUID_HOME/TO_BRIDGE_FROM_TDS   # this will be a symbolic link and used to simulate comms failure for data being uplinked to the ISS
COMMS_UP=_COMMS_UP   # directory name post_fix to append to enable symbolic link switching
COMMS_DOWN=_COMMS_DOWN   # directory name post_fix to append to enable symbolic link switching
REQUEST_PATH=request_path #open_fl pipeline splits uplink and downlink pipeline into response and request subdirectories
RESPONSE_PATH=response_path #open_fl pipeline splits uplink and downlink pipeline into response and request subdirectories

REQUIRED_DIRS="$TO_BRIDGE_FROM_COLABSHIM$COMMS_UP/$REQUEST_PATH $TO_BRIDGE_FROM_COLABSHIM$COMMS_UP/$RESPONSE_PATH $TO_BRIDGE_FROM_COLABSHIM$COMMS_DOWN/$REQUEST_PATH $TO_BRIDGE_FROM_COLABSHIM$COMMS_DOWN/$RESPONSE_PATH $TO_BRIDGE_FROM_TDS$COMMS_UP/$REQUEST_PATH $TO_BRIDGE_FROM_TDS$COMMS_UP/$RESPONSE_PATH $TO_BRIDGE_FROM_TDS$COMMS_DOWN/$REQUEST_PATH $TO_BRIDGE_FROM_TDS$COMMS_DOWN/$RESPONSE_PATH"
REQUIRED_LINKS="$TO_BRIDGE_FROM_TDS $TO_BRIDGE_FROM_COLABSHIM"

this_proc=$(basename $0 .sh)
OPTIONS="[loop delay in seconds] [install|uninstall|reset|retain|status|help]"


#file count function - one parameter is directory
fc() {
        echo $(ls $1 | wc -l 2>/dev/null)
}

# process CLI parameters
LOOP_SLEEP=${1:-10}
if [[ $LOOP_SLEEP =~ ^[\-0-9]+$ ]]; then   #if no paramters or 1st parm is numeric, then that 1st param is the loop delay value
        ACTION=${2:-reset}
else   # otherwise the 1st param is the action parameter and the second parameter (if there is one) is the loop delay value
        ACTION=$LOOP_SLEEP
        LOOP_SLEEP=${2:-10}
        if ! [[ $LOOP_SLEEP =~ ^[\-0-9]+$ ]]; then
                printf "Error - neither [$ACTION] nor [$LOOP_SLEEP] are valid delays (in seconds) for the file generation loop. Try [$this_proc help] for more information.\n"
                exit 1
        fi
fi

# Abort if this bash script does not have r/w/x access to the $FLUID_HOME directory, unless just asking for help message
if [ "$ACTION" != "help" ]; then
        if [ ! -r $FLUID_HOME ] || [ ! -w $FLUID_HOME ]; then
                printf "Error - $this_proc must be executed using an account that has read/write access to $FLUID_HOME\n"
                exit 1
        fi
fi

case $ACTION in

        install)
                ./comms_updown.sh install
                exit 0
                ;;

        uninstall)
                ./comms_updown.sh uninstall
                exit 0
                ;;

        status)
                ./comms_updown.sh status
                exit 0
                ;;

        reset)
                printf "\nResetting the test environment and then running new file generation process...\n"
                ./comms_updown.sh uninstall   # clear all links and directories
                ./comms_updown.sh install          # fresh reinstall
                ;;

        retain)
                printf "\nContinue running the file generation process without resetting the test environment...\n"
                ;;

        help)
                printf "\n usage: $this_proc [n] [$OPTIONS]\n"
                printf "        n = positive integer, number of seconds that $this_proc should delay before looping to creat another file.\n"
                printf "        install = configure directories and symbolic links to use $this_proc and the comms_updown utility.\n"
                printf "        uninstall = remove $this_proc environment and restore the normal FLUID configuration.\n"
                printf "        reset [DEFAULT] = clear all files and links from previous tests before starting a new $this_proc file generation process\n"
                printf "        retain = start a new $this_proc file generation process without clearing files and links from previous tests\n"
                printf "        status = display the current status of the communications (ie. up or down) and the number of buffered files.\n"
                printf "        help = this message.\n\n"
                exit 0
                ;;

        *)      # invalid action parameter
                printf "Error - invalid action parameter.\n"
                printf "usage: $this_proc $OPTIONS\n"
                exit 1
                ;;
esac

# check to make sure the FLUID comms testing environment has been installed and is properly properly configured
MISSING_LINKS=" "
MISSING_DIRS=" "
for fluid_dir in $REQUIRED_DIRS; do
        if [ ! -d $fluid_dir ]; then
                MISSING_DIRS+="\n$fluid_dir"
        fi
done
for fluid_link in $REQUIRED_LINKS; do
        if [ ! -L $fluid_link ]; then
                MISSING_LINKS+="\n$fluid_link"
        fi
done
if [ "$MISSING_DIRS" != " " ]; then
        printf "Error - The following directories are missing:$MISSING_DIRS\n"
fi
if [ "$MISSING_LINKS" != " " ]; then
        printf "Error - The following symbolic links are missing:$MISSING_LINKS\n"
fi
if [ "$MISSING_DIRS" != " " -o "$MISSING_LINKS" != " " ]; then
        printf "$this_proc is a test harness for the comms_updown.sh script. You must first install that environment by running [$this_proc install].\n"
        exit 1
fi


printf "\n>>> Starting simulated FLUID file generation at $LOOP_SLEEP second intervals. Press any key to stop.\n\n"
printf "FILE COUNTS:  -------FROM_COLABSHIM--------                 ---------FROM_TDS--------\n"
printf "          --COMMS_UP--            --COMMS_DOWN--        --COMMS_UP--         --COMMS_DOWN--\n"
printf "          REQ      RES            REQ        RES        REQ      RES         REQ        RES\n"
printf "          ---      ---            ---        ---        ---      ---         ---        ---\n"

while true; do
        TIME=$(date "+%T")
        touch $TO_BRIDGE_FROM_COLABSHIM/$REQUEST_PATH/COLABSHIM_TESTFILE_$TIME
        touch $TO_BRIDGE_FROM_COLABSHIM/$RESPONSE_PATH/COLABSHIM_TESTFILE_$TIME
        touch $TO_BRIDGE_FROM_TDS/$REQUEST_PATH/TDS_TESTFILE_$TIME
        touch $TO_BRIDGE_FROM_TDS/$RESPONSE_PATH/TDS_TESTFILE_$TIME
        if read -t 0.01 -N 1; then
                printf "\n\nEXITING...\n"
                break
        else
printf "          %0*u      %0*u            %0*u        %0*u        %0*u      %0*u         %0*u        %0*u\r" \
                3 $(fc $TO_BRIDGE_FROM_COLABSHIM$COMMS_UP/$REQUEST_PATH) \
                3 $(fc $TO_BRIDGE_FROM_COLABSHIM$COMMS_UP/$RESPONSE_PATH) \
                3 $(fc $TO_BRIDGE_FROM_COLABSHIM$COMMS_DOWN/$REQUEST_PATH) \
                3 $(fc $TO_BRIDGE_FROM_COLABSHIM$COMMS_DOWN/$RESPONSE_PATH) \
                3 $(fc $TO_BRIDGE_FROM_TDS$COMMS_UP/$REQUEST_PATH) \
                3 $(fc $TO_BRIDGE_FROM_TDS$COMMS_UP/$RESPONSE_PATH) \
                3 $(fc $TO_BRIDGE_FROM_TDS$COMMS_DOWN/$REQUEST_PATH) \
                3 $(fc $TO_BRIDGE_FROM_TDS$COMMS_DOWN/$RESPONSE_PATH)
        fi
        sleep $LOOP_SLEEP
done

printf ">>> FLUID file generation halted.\n\n"
exit 0
