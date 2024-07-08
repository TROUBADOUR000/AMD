root_path_name=../data/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

seed=2024

seq_len=512

for pred_len in 96 192 336 720
do
  python -u main.py \
    --seed $seed \
    --data $root_path_name$data_path_name \
    --feature_type M \
    --target OT \
    --checkpoint_dir ./checkpoints \
    --name $model_id_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --n_block 1 \
    --alpha 1.0 \
    --mix_layer_num 3 \
    --mix_layer_scale 2 \
    --patch 4 \
    --norm False \
    --layernorm False \
    --dropout 0.1 \
    --train_epochs 10 \
    --batch_size 128 \
    --learning_rate 0.00005 \
    --result_path result.csv
done
