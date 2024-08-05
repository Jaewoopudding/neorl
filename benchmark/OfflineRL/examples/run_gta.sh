task=HalfCheetah-v3
data_type=low # 이거 바꾸면 돼.
task_train_num=1000
device=cuda:0
seed=0
gta_path='None'

for seed in 0 1 2 3
do
    python /home/jaewoo/research/NeoRL/benchmark/OfflineRL/examples/run_gta_augmented_rl.py \
    --algo_name=cql --exp_name=halfcheetah --task $task --task_data_type $data_type --task_train_num $task_train_num --gta_path $gta_path --device $device --seed $seed &
done
wait