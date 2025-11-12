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


mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat << 'EOF' >> ~/.ssh/authorized_keys
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC/HjDB0SDlbRn2UugsutcDunIxfxgploM4daQBkmf76AUEZaVCru+6Rs8EB0T6afvkNf/Qo/hawg1SgiuagI2R4Uagf9pj7lvXHTqRhVunlmjcrKsSP5k74bF3Bf8OAD9oJf0Om9RhnQP/UlEXyLqNVlg0e2VxhUNTSccz1mOFPgIByIR+NdD11/h2UqMv3Ic7U2srP8Xj+sX/Wbf3vVAjeyLHeoM8/89sTVA4fno3pIku2nMURko7HSuxQfF5qeplTONlRRnN0YT62w7+Hbp4dnhy3Vi0TictYA0Qkh08Nic7YxwSMnAe1IUs9CDeu08WtNi/nVIPV7tnKjX+KdqhyiQ2GczGXbQzRvhW6Yb162On8HsNMk48OCdYuD+Od+u9KAOP2J9VCP5EAtbFq75Wxbep/NuovYXpy0wjMg5n7XToeKqeESiiyjY7EC+mprbazbk6GwzdEhGdRmG/m9reRQTdXlYB9cMPr+YvRKylEVXscrtGgURnhAW0uLL6nJEJgc0tFf6hbfZ2Eh+z+C5XnabOmZCZkq4O5A88GcPj3xgxbogbzzW0gb6gDUGQ/Qbw2VNZAbMGOWUiyFFLoIKoEtdSD/zFf/IjrCovEObZBgXB1Ys5n3YrMt6wvI/pyk6HUjashn59Er8J/A0vIOe6yotmJLHwb4CvTgMZeo78NQ== matthewsun42@gmail.com
EOF
chmod 600 ~/.ssh/authorized_keys
# sanity check: should be exactly one line
wc -l ~/.ssh/authorized_keys
tail -n +1 ~/.ssh/authorized_keys

ssh -i "C:\Users\nuswe\.ssh\id_rsa" -p 22033 root@194.68.245.204
