for i in {0..14}; do
    CUDA_VISIBLE_DEVICES='' python generate_episodes.py --shard $i --num_episodes 1000 --save_steps 50 &
done
wait
python merge_shard.py