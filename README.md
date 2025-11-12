# skribbl

to run test_dataloader.py:

python test_dataloader.py \
  --manifest training_manifest.csv \
  --class_index class_index.json \
  --prompts prompts.json \
  --batch_size 64 \
  --num_workers 4 \
  --image_size 224 \
  --limit 1024 \
  --shuffle_first  # if you added this flag


training:

python train_clip.py \
  --manifest training_manifest.csv \
  --class_index class_index.json \
  --prompts prompts.json \
  --splits splits.csv \
  --data_root . \
  --image_size 224 \
  --batch_size 256 \
  --accum_steps 1 \
  --lr 5e-5 \
  --epochs 10 \
  --model ViT-B-16 \
  --precision amp \
  --workers 8 \
  --out_dir checkpoints


eval:

python eval_clip.py \
  --manifest training_manifest.csv \
  --class_index class_index.json \
  --prompts prompts.json \
  --splits splits.csv \
  --data_root . \
  --image_size 224 \
  --batch_size 256 \
  --workers 8 \
  --checkpoint checkpoints/last.ckpt \
  --save_csv eval_results.csv \
  --letter_filter  # enables the letter-aware evaluation


