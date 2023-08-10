LR=(0.1 0.5)
weights=(0.3 0.5 1.0)
temps=(0.5 0.3 0.1)
temp_dists=(0.5 0.3 0.1)
Buffersize=200
seed=123
for lr in "${LR[@]}"
do
  for weight in "${weights[@]}"
  do
    for temp in "${temps[@]}"
    do
      for temp_dist in "${temp_dists[@]}"
      do
        output=mlp_rmnist_buffer${Buffersize}_lr${lr}_weight${weight}_temp${temp}_temp_dis${temp_dist}_seed${seed}
        if [ -e "rmnistlog/"$output ]; then
          echo "exit"$output >>rmnistlog/output
        else
          python mlp_rmnist_main.py --feat_dim 128 --num_workers 8 --dataset rmnist --batch_size 1024 --epochs 30 --start_epoch 50 --seed $seed --Buffersize $Buffersize --learning_rate $lr --weighted_loss $weight --temp $temp --temp_dist $temp_dist >>rmnistlog/$output
        fi
      done
    done
  done
done




