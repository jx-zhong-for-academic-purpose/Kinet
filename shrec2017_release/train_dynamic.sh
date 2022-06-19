command_file=`basename "$0"`
gpu=0
model=model_cls_dynamic
num_point=128
num_frame=32
batch_size=8
learning_rate=0.001
model_path=trained_static_model/model.ckpt
log_dir=log_${model}_frames${num_frame}_batch_size${batch_size}
modality=dynamic

python main.py --phase=train --work-dir=${log_dir}/results/ \
    --modality $modality \
    --gpu $gpu \
    --network_file $model \
    --learning_rate $learning_rate \
    --model_path $model_path \
    --log_dir $log_dir \
    --num_point $num_point \
    --num_frame $num_frame \
    --batch_size $batch_size \
    --command_file $command_file \
    > $log_dir.txt 2>&1 &
