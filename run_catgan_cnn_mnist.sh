EXEC_SCRIPT=/lfs/local/0/annhe/catgan/catgan_cnn.py

#TAN_PATH='/dfs/scratch0/annhe/experiments/ddsm/log/2018_02_24/tan_only_mammo_paper_tfs36_debug_23_19_39/tan/'

OUTPUT_PATH="/dfs/scratch0/annhe/experiments/catgan/logs"
#DATA_DIR="/dfs/scratch0/annhe/tanda_750_90_10_split/"
#LABEL_JSON="$DATA_DIR/mass_to_label.json"
EXP_NAME='catgan_exp'

START_DATE=`date +"%m_%d_%y"`
RUN_LOG_PATH="$OUTPUT_PATH/${START_DATE}/${EXP_NAME}/${TRIAL_NAME}/run_log.txt"

export CUDA_VISIBLE_DEVICES=0


for i in 0
do
for lr in 0.00001 0.0001 0.001
do
for dp in  0.1 
do
for bs in 32 
do
for epochs in 100
do
for dec in 0.01 
do
TRIAL_NAME=${EXP_NAME}_ep_${epochs}_wt_${weights}_lr_${lr}_do_${dp}_bs_${bs}
TIME=`date +"%H_%M_%S"`
LOG_PATH="$OUTPUT_PATH/${START_DATE}/${EXP_NAME}/${TRIAL_NAME}/"
mkdir -p $LOG_PATH
LOGFILE="$LOG_PATH/terminal_run_log_${TIME}.log"
python $EXEC_SCRIPT --dir_name LOG_PATH 2>&1 | tee $LOGFILE
done
done
done
done
done
done

