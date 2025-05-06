#!/bin/bash

# Load crisp working directories and other runtim configurations
source /scripts/crisp-config.sh

process_args() {
	ARGS=$1
	for arg in $ARGS
	do
		case $arg in
			-r |--role)
			ROLE="$2"
        		shift # Remove argument name from processing
        		shift # Remove argument value from processing
			;;

       			*)
			echo "wrong usage: $ARGS"
			exit 1
       			;;
		esac
	done	
	echo $ROLE
}

update_plans_iss() {
	ROLE=$1
	# update plan.yaml
	/usr/bin/sed -i "s/.*agg_addr.*/    agg_addr: $AGG_ISS_HOST/" ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i "s/.*agg_port.*/    agg_port: $AGG_ISS_PORT/" ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i "s/.*disable_client_auth.*/    disable_client_auth: true/" ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i 's/.*disable_tls.*/    disable_tls: true/' ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i "s/.*collaborator_count.*/    collaborator_count: $COLABORATOR_COUNT/" ~/crisp/fl_plan/plan.yaml
	#/usr/bin/sed -i "s/.*collaborator_count.*/    collaborator_count: 1/" ~/crisp/fl_plan/plan.yaml
	#/usr/bin/sed -i "s/.*best_state_path.*/    best_state_path: $WORKSPACE_ISS_AGG_DIR\/crisp_best_.pbuf/" ~/crisp/fl_plan/plan.yaml
	#/usr/bin/sed -i "s/.*init_state_path.*/    init_state_path: $WORKSPACE_ISS_AGG_DIR\/crisp_init_.pbuf/" ~/crisp/fl_plan/plan.yaml
	#/usr/bin/sed -i "s/.*last_state_path.*/    last_state_path: $WORKSPACE_ISS_AGG_DIR\/crisp_last_.pbuf/" ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i "s/.*rounds_to_train.*/    rounds_to_train: $ROUNDS/" ~/crisp/fl_plan/plan.yaml
        /usr/bin/sed -i "s/.*db_store_rounds.*/    db_store_rounds: $ROUNDS/" ~/crisp/fl_plan/plan.yaml
        /usr/bin/sed -i "s/.*num_epochs.*/      num_epochs: $EPOCHS/" ~/crisp/fl_plan/plan.yaml
	
	cat ~/crisp/fl_plan/plan.yaml

	#echo "colab-earth,/data/col_0" > ~/crisp/fl_plan/data.yaml
	#echo "colab-iss,/data/col_1" >> ~/crisp/fl_plan/data.yaml

	cat /dev/null > ~/crisp/fl_plan/data.yaml
	i=0	
	for colab in $COLABS
	do	
		echo "${colab},/data/col_${i}" >> ~/crisp/fl_plan/data.yaml
		((i=$i+1))
	done

	cat ~/crisp/fl_plan/data.yaml

	#echo "collaborators:" > ~/crisp/fl_plan/cols.yaml	
	#echo "- colab-earth" >> ~/crisp/fl_plan/cols.yaml	
	#echo "- colab-iss" >> ~/crisp/fl_plan/cols.yaml	
	echo "collaborators:" > ~/crisp/fl_plan/cols.yaml	
	for colab in $COLABS
	do
		echo "- ${colab}" >> ~/crisp/fl_plan/cols.yaml	
	done

	cat ~/crisp/fl_plan/cols.yaml

	
	if [ "$ROLE" == "agg-iss" ]
	then
		cd $WORKSPACE_ISS_AGG_DIR 
	else
		cd $WORKSPACE_ISS_COLAB_DIR
	fi
	rm -rf workspace 
	fx workspace create --prefix workspace --template torch_cnn_mnist 
	chmod 777 workspace
	cd workspace 

	cp -r ~/crisp/fl_plan/* plan/
	cp -r ~/crisp/fl_src/* src/
	fx plan initialize -a $AGG_ISS_HOST
	#fx plan initialize
}	

update_plans_earth() {
	ROLE=$1
	# update plan.yaml
	/usr/bin/sed -i "s/.*agg_addr.*/    agg_addr: $AGG_EARTH_HOST/" ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i "s/.*agg_port.*/    agg_port: $AGG_EARTH_PORT/" ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i "s/.*disable_client_auth.*/    disable_client_auth: true/" ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i 's/.*disable_tls.*/    disable_tls: true/' ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i "s/.*collaborator_count.*/    collaborator_count: $COLABORATOR_COUNT/" ~/crisp/fl_plan/plan.yaml
	#/usr/bin/sed -i "s/.*collaborator_count.*/    collaborator_count: 1/" ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i "s/.*best_state_path.*/    best_state_path: $WORKSPACE_EARTH_AGG_DIR\/crisp_best_.pbuf/" ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i "s/.*init_state_path.*/    init_state_path: $WORKSPACE_EARTH_AGG_DIR\/crisp_init_.pbuf/" ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i "s/.*last_state_path.*/    last_state_path: $WORKSPACE_EARTH_AGG_DIR\/crisp_last_.pbuf/" ~/crisp/fl_plan/plan.yaml
	/usr/bin/sed -i "s/.*rounds_to_train.*/    rounds_to_train: $ROUNDS/" ~/crisp/fl_plan/plan.yaml
        /usr/bin/sed -i "s/.*db_store_rounds.*/    db_store_rounds: $ROUNDS/" ~/crisp/fl_plan/plan.yaml
        /usr/bin/sed -i "s/.*num_epochs.*/      num_epochs: $EPOCHS/" ~/crisp/fl_plan/plan.yaml
	
	cat ~/crisp/fl_plan/plan.yaml


	#echo "colab-earth,/data/col_0" > ~/crisp/fl_plan/data.yaml
	#echo "colab-iss,/data/col_1" >> ~/crisp/fl_plan/data.yaml

	cat /dev/null > ~/crisp/fl_plan/data.yaml
	i=0	
	for colab in $COLABS
	do	
		echo "${colab},/data/col_${i}" >> ~/crisp/fl_plan/data.yaml
		((i=$i+1))
	done

	cat ~/crisp/fl_plan/data.yaml

	#echo "collaborators:" > ~/crisp/fl_plan/cols.yaml	
	#echo "- colab-earth" >> ~/crisp/fl_plan/cols.yaml	
	#echo "- colab-iss" >> ~/crisp/fl_plan/cols.yaml	
	echo "collaborators:" > ~/crisp/fl_plan/cols.yaml	
	for colab in $COLABS
	do
		echo "- ${colab}" >> ~/crisp/fl_plan/cols.yaml	
	done

	cat ~/crisp/fl_plan/cols.yaml

	if [ "$ROLE" == "agg-earth" ]
        then
                cd $WORKSPACE_EARTH_AGG_DIR
        elif [ "$ROLE" == "colab-earth" ]
	then
                cd $WORKSPACE_EARTH_COLAB_REAL_DIR
	elif [ "$ROLE" == "colab-iss" ]
	then
		cd $WORKSPACE_EARTH_COLAB_SHIM_DIR
	else
		echo wrong role:  $ROLE
		exit 1
        fi

	rm -rf workspace
	fx workspace create --prefix workspace --template torch_cnn_mnist 
	chmod 777 workspace
	cd workspace 

	cp -r ~/crisp/fl_plan/* plan/
	cp -r ~/crisp/fl_src/* src/
	fx plan initialize -a $AGG_EARTH_HOST
}

activate_conda() {
	VENV=$1
	echo "activating conda env"
	conda init bash
	. ~/.bash_profile > /dev/null
	. ~/.bashrc > /dev/null
	conda activate $VENV 
}

run_agg_iss() {
	ROLE=agg-iss
	if [ -d $WORKSPACE_ISS_AGG_DIR ]
	then
		rm -rf $WORKSPACE_ISS_AGG_DIR
	fi
	mkdir -p $WORKSPACE_ISS_AGG_DIR
	update_plans_iss agg-iss 
	fx aggregator shim
}

run_agg_earth() {
	ROLE=agg-earth
	if [ -d $WORKSPACE_EARTH_AGG_DIR ]
	then
		rm -rf $WORKSPACE_EARTH_AGG_DIR
	fi
	mkdir -p $WORKSPACE_EARTH_AGG_DIR
	update_plans_earth agg-earth 
	fx aggregator start
	# now wait for job to be done and then get results ready 
	while [ ! -f ${WORKSPACE_EARTH_AGG_DIR}/workspace/save/crisp_best_.pbuf ]
	do
		sleep 5
	done
	cd ${WORKSPACE_EARTH_AGG_DIR}/workspace/
	fx model save -m save/crisp_best_.pbuf
	sleep 10
		
}

run_colab_iss() {
	ROLE=colab-iss
	if [ -d $WORKSPACE_ISS_COLAB_DIR ]
	then
		rm -rf $WORKSPACE_ISS_COLAB_DIR
	fi
	mkdir -p $WORKSPACE_ISS_COLAB_DIR
	update_plans_iss colab-iss
	fx collaborator start -n $ROLE -p plan/plan.yaml -d plan/data.yaml
}

run_colab_shim_earth() {
	ROLE=colab-iss
	if [ -d $WORKSPACE_EARTH_COLAB_SHIM_DIR ]
	then
		rm -rf $WORKSPACE_EARTH_COLAB_SHIM_DIR
	fi
	mkdir -p $WORKSPACE_EARTH_COLAB_SHIM_DIR
	update_plans_earth $ROLE 
	mkdir $WORKSPACE_EARTH_COLAB_SHIM_DIR/workspace/proto_path
	fx collaborator shim  -n $ROLE -p plan/plan.yaml -d plan/data.yaml
}

run_colab_real_earth() {
	ROLE=colab-earth
	if [ -d $WORKSPACE_EARTH_COLAB_REAL_DIR ]
	then
		rm -rf $WORKSPACE_EARTH_COLAB_REAL_DIR
	fi
	mkdir -p $WORKSPACE_EARTH_COLAB_REAL_DIR
	update_plans_earth $ROLE 
	fx collaborator start  -n $ROLE -p plan/plan.yaml -d plan/data.yaml
}
	

main() {
	activate_conda venv_3.7
	ROLE=$(process_args $@)
	HOST=$(echo $ROLE | cut -d- -f1)
	LOC=$(echo $ROLE | cut -d- -f2)
	echo my role in runCrisp: $ROLE
	case "$ROLE" in
		agg-iss)
			run_agg_iss
			;;
	
		agg-earth)
			run_agg_earth	
			;;
	
		colab-iss)
			run_colab_iss
			;;
		colab-earth)
			run_colab_real_earth
			;;
		colab-shim)
			run_colab_shim_earth 
			;;
		*)
			echo "wrong usage: $ROLE"
			exit 1
			;;
	
	esac
}

main $@ 
