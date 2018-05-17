EXEC_SCRIPT=/lfs/1/annhe/catgan/train_catgan_model.py

#TAN_PATH='/dfs/scratch0/annhe/experiments/ddsm/log/2018_02_24/tan_only_mammo_paper_tfs36_debug_23_19_39/tan/'

OUTPUT_PATH="/dfs/scratch0/annhe/experiments/catgan/logs"
#DATA_DIR="/dfs/scratch0/annhe/tanda_750_90_10_split/"
#LABEL_JSON="$DATA_DIR/mass_to_label.json"
EXP_NAME='catgan_model'
DEVICE_NAME='cpu'
START_DATE=`date +"%m_%d_%y"`
RUN_LOG_PATH="$OUTPUT_PATH/${START_DATE}/${EXP_NAME}/${TRIAL_NAME}/run_log.txt"

CUDA_VISIBLE_DEVICES=0


for i in 0
do
for lr in 0.001
do
for dp in  0.1 
do
for bs in 48 
do
for ce in 1.0
do
for epochs in 25
do
TRIAL_NAME=${EXP_NAME}_ep_${epochs}_wt_${weights}_lr_${lr}_do_${dp}_bs_${bs}_ce_${ce}
TIME=`date +"%H_%M_%S"`
LOG_PATH="$OUTPUT_PATH/${START_DATE}/${EXP_NAME}/${TRIAL_NAME}/${TIME}/"
mkdir -p $LOG_PATH
LOGFILE="$LOG_PATH/terminal_run_log_${TIME}.log"
python $EXEC_SCRIPT --device_name $DEVICE_NAME --batch_size $bs --lr $lr --num_epochs $epochs --z_dim 500 --n_class 2 --ce_term $ce --dir_name $LOG_PATH 2>&1 | tee $LOGFILE
done
done
done
done
done
done


