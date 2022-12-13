for i in {0..14}; do
    python generate_episodes.py --shard $i --num_episodes 600 &
done
wait
python merge_shard.py