#!/bin/bash

# Simulates communications breaks by redirecting and holding data files on the bridge node into a buffered directory during comms-down periods.

# FLUID_HOME=/home/fluid     # Use this to enforce that this script can ONLY run under the "fluid" user account
FLUID_HOME=~    # Use this to test this script under any user account (e.g. ec2-user)

TO_BRIDGE_FROM_COLABSHIM=$FLUID_HOME/TO_BRIDGE_FROM_COLABSHIM   # this will be a symbolic link and used to simulate comms failure for data being downlinked from the ISS
TO_BRIDGE_FROM_TDS=$FLUID_HOME/TO_BRIDGE_FROM_TDS   # this will be a symbolic link and used to simulate comms failure for data being uplinked to the ISS
COMMS_UP=_COMMS_UP   # directory name post_fix to append to enable symbolic link switching
COMMS_DOWN=_COMMS_DOWN   # directory name post_fix to append to enable symbolic link switching
REQUEST_PATH=request_path #open_fl pipeline splits uplink and downlink pipeline into response and request subdirectories
RESPONSE_PATH=response_path #open_fl pipeline splits uplink and downlink pipeline into response and request subdirectories
DISABLED=_DISABLED   # directory postfix to temporarily rename existing FLUID directories so that they will not duplicate/conflict with symbolic link names created in this script.

REQUIRED_DIRS="$TO_BRIDGE_FROM_COLABSHIM$COMMS_UP $TO_BRIDGE_FROM_COLABSHIM$COMMS_DOWN $TO_BRIDGE_FROM_TDS$COMMS_UP $TO_BRIDGE_FROM_TDS$COMMS_DOWN \
$TO_BRIDGE_FROM_COLABSHIM$COMMS_UP/$REQUEST_PATH $TO_BRIDGE_FROM_COLABSHIM$COMMS_UP/$RESPONSE_PATH $TO_BRIDGE_FROM_COLABSHIM$COMMS_DOWN/$REQUEST_PATH \
$TO_BRIDGE_FROM_COLABSHIM$COMMS_DOWN/$RESPONSE_PATH $TO_BRIDGE_FROM_TDS$COMMS_UP/$REQUEST_PATH $TO_BRIDGE_FROM_TDS$COMMS_UP/$\
RESPONSE_PATH $TO_BRIDGE_FROM_TDS$COMMS_DOWN/$REQUEST_PATH $TO_BRIDGE_FROM_TDS$COMMS_DOWN/$RESPONSE_PATH"

REQUIRED_LINKS="$TO_BRIDGE_FROM_TDS $TO_BRIDGE_FROM_COLABSHIM"

OPTIONS="install|uninstall|up|down|toggle|status|help"
this_proc=$(basename $0 .sh)
conflicting_fluid_processes=$(pgrep -x 'fluidsynch|oursynch')

#file count function - one parameter is directory
fc() {
        echo $(ls $1 | wc -l 2>/dev/null)
}

# Get the ACTION parameter, or abort if no argument provided
ACTION="${1:?usage: $this_proc $OPTIONS}"

if [ "$ACTION" != "help" ]; then
        # Abort if this bash script does not have r/w/x access to the FLUID_HOME directory
        if [ ! -r $FLUID_HOME ] || [ ! -w $FLUID_HOME ]; then
                printf "Error - $this_proc must be executed using an account that has read/write access to $FLUID_HOME\n"
                exit 1
        fi
fi


case $ACTION in

        install)
                # -------- INSTALL ENVIRONMENT-------------
                # Abort install if a fluid sync process is currently active (e.g. fluidsynch.sh oursynch.sh, fluidtest_synch.sh)
                if [ -n "$conflicting_fluid_processes" ]; then
                        printf "Please terminate the following fluid process before installing $this_proc: [$conflicting_fluid_processes]\n"
                        # exit 1
                fi

                #Check with user if this is not the bridge node
                if [ $(hostname) != "bridge" ]; then
                        printf "Warning - $this_proc is intended be installed on the bridge node of the FLUID network.\n"
                        read -p "Do you wish to install this on [$(hostname)] ?" yn
                        case $yn in
                                [Yy]* ) printf "Installing $this_proc on [$(hostname)]...\n";;
                                [Nn]* ) exit 1;;
                                * ) printf "Please respond with yes or no.\n";;
                        esac
                fi

                # clean up symbolic links from any prior installation
                for fluid_link in $REQUIRED_LINKS; do
                        if [ -L $fluid_link ]; then
                                unlink $fluid_link
                        fi
                done

                # if of the required links already exist as physical directories then rename the directories
                # to avoid confusion with the symbolic links that will be created
                for fluid_link in $REQUIRED_LINKS; do
                        if [ -d $fluid_link ]; then
                                mv $fluid_link $fluid_link$DISABLED
                        fi
                done

                # create required directories if they don't yet exist
                for fluid_dir in $REQUIRED_DIRS; do mkdir -p $fluid_dir; done

                # create symbolic links with a starting state of "comms up"
                ln -s $TO_BRIDGE_FROM_TDS$COMMS_UP $TO_BRIDGE_FROM_TDS
                ln -s $TO_BRIDGE_FROM_COLABSHIM$COMMS_UP $TO_BRIDGE_FROM_COLABSHIM
                printf "$this_proc successfully installed\n"
                ;;

        uninstall)
                # -------- UNINSTALL ENVIRONMENT-------------
                # Abort uninstall if a fluid sync process is currently active (e.g. fluidsynch.sh oursynch.sh, fluidtest_synch.sh)
                if [ -n "$conflicting_fluid_processes" ]; then
                        printf "Please terminate the following fluid process before uninstalling $this_proc: [$conflicting_fluid_processes]\n"
                        # exit 1
                fi

                # remove symbolic links
                for fluid_link in $REQUIRED_LINKS; do
                        if [ -L $fluid_link ]; then
                                unlink $fluid_link
                        fi
                done

                # remove the special comms_up and comms_down directories
                for fluid_dir in $REQUIRED_DIRS; do rm -rf $fluid_dir; done

                # if previously existing directories were disabled during the install, then reinstate them.
                for restore in $(ls -d $FLUID_HOME/*$DISABLED  2>/dev/null); do mv $restore ${restore%$DISABLED}; done

                printf "$this_proc successfully uninstalled\n"
                ;;

        status)
                if [ ! -L $TO_BRIDGE_FROM_TDS ]; then
                        printf "Run $this_proc using the install option to configure the environment before attempting to display the communication state.\n"
                        exit 1
                else
                        if [ -z $(readlink $TO_BRIDGE_FROM_TDS | grep $COMMS_UP) ] ; then
                                COMM_STAT=DOWN
                        else
                                COMM_STAT=UP
                        fi
                        printf "\nCommunications have been $COMM_STAT for $(($(date +%s) - $(stat -c "%Y" $TO_BRIDGE_FROM_TDS))) seconds\n"
                        printf "NOMINAL file pipeline:  TO_BRIDGE_FROM_COLABSHIM        REQUEST=[$(fc $TO_BRIDGE_FROM_COLABSHIM$COMMS_UP/$REQUEST_PATH)]        RESPONSE=[$(fc $TO_BRIDGE_FROM_COLABSHIM$COMMS_UP/$RESPONSE_PATH)]\n"
                        printf "                        TO_BRIDGE_FROM_TDS              REQUEST=[$(fc $TO_BRIDGE_FROM_TDS$COMMS_UP/$REQUEST_PATH)]      RESPONSE=[$(fc $TO_BRIDGE_FROM_TDS$COMMS_UP/$RESPONSE_PATH)]\n\n"
                        printf "BUFFERED file pipeline: TO_BRIDGE_FROM_COLABSHIM        REQUEST=[$(fc $TO_BRIDGE_FROM_COLABSHIM$COMMS_DOWN/$REQUEST_PATH)]      RESPONSE=[$(fc $TO_BRIDGE_FROM_COLABSHIM$COMMS_DOWN/$RESPONSE_PATH)]\n"
                        printf "                        TO_BRIDGE_FROM_TDS              REQUEST=[$(fc $TO_BRIDGE_FROM_TDS$COMMS_DOWN/$REQUEST_PATH)]    RESPONSE=[$(fc $TO_BRIDGE_FROM_TDS$COMMS_DOWN/$RESPONSE_PATH)]\n\n"
                fi
                ;;

        up | down | toggle)
                # -------- COMMS UP/DOWN STATE CHANGE ----------
                # check to make sure environment has been installed and is properly properly configured
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
                        printf "Run $this_proc using the install option to configure the environment before attempting to modify the communication state.\n"
                        exit 1
                fi

                case $ACTION in
                        toggle)
                                if [ -z $(readlink $TO_BRIDGE_FROM_TDS | grep $COMMS_UP) ] ; then
                                        NEW_STATE=$COMMS_UP
                                else
                                        NEW_STATE=$COMMS_DOWN
                                fi
                                ;;
                        up)
                                NEW_STATE=$COMMS_UP
                                ;;
                        down)
                                NEW_STATE=$COMMS_DOWN
                                ;;
                esac

                # set symbolic link to the new state
                unlink $TO_BRIDGE_FROM_TDS
                unlink $TO_BRIDGE_FROM_COLABSHIM
                ln -s $TO_BRIDGE_FROM_TDS$NEW_STATE $TO_BRIDGE_FROM_TDS
                ln -s $TO_BRIDGE_FROM_COLABSHIM$NEW_STATE $TO_BRIDGE_FROM_COLABSHIM

                # If changing state from comms_down to comms-up then loop to move any files that accumulated in the "COMMS_DOWN" buffer back over to normal "COMMS_UP" directory
                # The file move loop occurs with a one second sleep to emulate the more gradual uplink/downlink of the file backlog that would have built up during
                # an actual communications outage, versus a single 'mv' to move all files in subsecond time, which does not emulate ISS comms latency.
                if [ $NEW_STATE == $COMMS_UP ]; then
                        for directory in $TO_BRIDGE_FROM_TDS~COMMS_STATUS~/$REQUEST_PATH $TO_BRIDGE_FROM_TDS~COMMS_STATUS~/$RESPONSE_PATH $TO_BRIDGE_FROM_COLABSHIM~COMMS_STATUS~/$REQUEST_PATH $TO_BRIDGE_FROM_COLABSHIM~COMMS_STATUS~/$RESPONSE_PATH; do
                                from_dir=${directory/~COMMS_STATUS~/_COMMS_DOWN}
                                to_dir=${directory/~COMMS_STATUS~/_COMMS_UP}
                                printf "\n"
                                for file in $(ls $from_dir); do
                                        printf "Processing $from_dir... Moving $file COMMS_UP\r"
                                        mv $from_dir/$file $to_dir
                                        sleep 1
                                done
                        done
                        printf "All comms_down buffered files have been moved to their nominal pipeline directories. \n"
                fi

                printf "FLUID communications now set to $NEW_STATE at $(date '+%T')\n"
                ;;

        help)
                printf "\n usage: $this_proc $OPTIONS\n"
                printf "        install = configure directories and symbolic links to use this $this_proc utility.\n"
                printf "        uninstall = remove $this_proc environment and restore the normal FLUID configuration.\n"
                printf "        up = set communications state to UP and enable normal movement of the FLUID data pipeline,\n\t\tand move files that were cached during a comms-down period back into the pipeline for normal processing.\n"
                printf "        down = set the communications state to DOWN and stop the flow of files in the FLUID data pipeline,\n\t\tcaching files for later processing once communications is restored.\n"
                printf "        toggle = switch the communications to alternative state (UP or DOWN).\n"
                printf "        status = display the current status of the communications (ie. up or down)\n"
                printf "        help = this message.\n\n"
                ;;

        *)      # invalid action parameter
                printf "Error - invalid action parameter\n"
                printf "usage: $this_proc $OPTIONS\n"
                exit 1
                ;;
esac
exit 0