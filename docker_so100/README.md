## TELEOPERATION - TEST
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=teleoperate
```

## RECORD DATASET
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a block" \
  --control.repo_id=${HF_USER}/so100_test \
  --control.tags='["so100","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=20 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=false \
  --control.display_cameras=false \
  --control.play_sounds=false
```

## VISUALIZE DATASET
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/so100_test
```
## PLAY EPISODE
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/so100_test \
  --control.episode=0
```

## TRAIN POLICY
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/so100_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so100_test \
  --job_name=act_so100_test \
  --policy.device=cuda \
  --wandb.enable=true
```

## RESUME TRAINING
```bash
python lerobot/scripts/train.py \
  --config_path=outputs/train/act_so100_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

## EVALUATE POLICY
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/eval_act_so100_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_so100_test/checkpoints/last/pretrained_model
```