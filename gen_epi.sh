for i in {0..14}; do
    CUDA_VISIBLE_DEVICES='' python generate_episodes.py --shard $i --num_episodes 20 --save_steps 5 --do_competition &
done
wait
#python merge_shard.py