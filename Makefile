.PHONY: build run down shell record

# Default values
ROBOT ?= franka
# Dynamically select the blueprint based on the active robot
BLUEPRINT ?= /assets/rerun/blueprints/$(ROBOT)_blueprint.rbl

# Dynamically select the container service based on ROBOT
SERVICE := $(if $(filter franka,$(ROBOT)),franka,tiago)

# Define VERBOSE as empty by default
VERBOSE ?=

# Default to demo_0 if not specified
DEMO ?= demo_0

record:
ifndef TASK
	$(error TASK is undefined. Usage: make record ROBOT=franka TASK=my_task_name)
endif
	docker compose exec -w /postprocess $(SERVICE) bash -c \
	"source /opt/ros/humble/setup.bash && \
	python3 /scripts/record.py --robot $(ROBOT) --task $(TASK)"\

convert-hdf5:
ifndef TASK
	$(error TASK is undefined. Usage: make convert-hdf5 ROBOT=franka TASK=one_bag_gray)
endif
	docker compose exec -w /postprocess $(SERVICE) bash -c \
	"source /opt/ros/humble/setup.bash && \
	python3 inria_franka_rosbag_to_hdf5.py \
		--rosbag_folder /datasets \
		--hdf5_dir /datasets/hdf5_converted \
		--config /postprocess/config_rosbag2hdf5/config.yaml \
		--tasks $(TASK) \
		$(if $(VERBOSE),--verbose)"

build:
	UID=$$(id -u) GID=$$(id -g) docker compose build

# Start specific robot (or both if ROBOT is empty)

run:
	xhost +local:docker
	docker compose up -d $(ROBOT)


down:
	docker compose down


shell:
	docker compose exec $(SERVICE) /bin/bash


logs:
	docker compose logs -f $(SERVICE)

visualize:
ifndef TASK
	$(error TASK is undefined. Usage: make visualize ROBOT=franka TASK=one_bag_gray)
endif
	docker compose exec -w /scripts $(SERVICE) bash -c \
	"source /opt/ros/humble/setup.bash && \
	python3 vis_dataset.py \
		--dataset /datasets/hdf5_converted/$(TASK).h5 \
		--demo $(DEMO) \
		--blueprint $(BLUEPRINT) \
		--output /assets/rerun/converted_datasets/$(TASK).rrd"