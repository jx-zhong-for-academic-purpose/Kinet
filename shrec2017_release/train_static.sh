command_file=`basename "$0"`
gpu=0
model=model_cls_static
num_point=128
num_frame=32
batch_size=8
learning_rate=0.001
log_dir=log_${model}_frames${num_frame}_batch_size${batch_size}
modality=static

#model_path=pretrained_on_modelnet/model.ckpt
#python main.py --phase=train --work-dir=${log_dir}/results/ --model_path model_path \
python main.py --phase=train --work-dir=${log_dir}/results/ \
    --modality $modality \
    --gpu $gpu \
    --network_file $model \
    --learning_rate $learning_rate \
    --log_dir $log_dir \
    --num_point $num_point \
    --num_frame $num_frame \
    --batch_size $batch_size \
    --command_file $command_file \
    > $log_dir.txt 2>&1 &
