## INSTALL THE LOCAL FILES COPIED TO THE DOCKER
```bash
pip install -e ".[feetech]"
```

## CHECK USB PORTS
```bash
python lerobot/scripts/find_motors_bus_port.py
```

Once you find the USB port, you can set it in the config file: ```class: So100RobotConfig```. By default, it is set to ```/dev/ttyACM0``` the leader arm and ```/dev/ttyACM1``` for the follower arm. Also check that the cameras are set to the correct ID ports, by default they are set into 8 and 10.

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
  --control.single_task="Grasp a block and put in a bowl" \
  --control.repo_id=${HF_USER}/so100_test \
  --control.tags='["so100","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=6 \
  --control.reset_time_s=5 \
  --control.num_episodes=20 \
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
  --control.episode=0 \
  --control.play_sounds=false
```

## TRAIN POLICY
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/so100_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so100_test \
  --job_name=act_so100_test \
  --policy.device=cuda \
  --wandb.enable=false
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
  --control.push_to_hub=false \
  --control.play_sounds=false \
  --control.display_cameras=false \
  --control.policy.path=outputs/train/act_so100_test/checkpoints/last/pretrained_model
```