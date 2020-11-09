#!/bin/bash

log_folder="./log_output"
if [ ! -x $log_folder ]; then
	mkdir $log_folder
fi

dataset_base_path="/home/fangxin/bs/data/openmldata/"


for arg in $*
do
	cur_time="`date +%Y-%m-%d-%H-%M-%S`"
	log_file="$log_folder/$arg-$cur_time.log"

#	model="LogisticRegression"
#	model="RandomForestClassifier"
	model="GBDTClassifier"

	python_command="python ../test/openml_model_selection_boasf.py --dataset_base_path=$dataset_base_path --dataset_idx=475 --time_budget=$arg 2>&1"
#	python_command="python ../test/openml_hyperparameter_search_boasf.py --model=$model --dataset_base_path=$dataset_base_path --dataset_idx=475 --time_budget=$arg 2>&1"

	log_command="tee -i $log_file"

	echo "Current time: $cur_time"
	echo "Run command: $python_command"
	echo "Log info into file: $log_file"
	eval "$python_command | $log_command"
#	eval "$python_command"
done
