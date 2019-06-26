#!/usr/bin/env bash

expid="mm45_v4_SqnoiseShoulder_exptownv9v10_mergefollowstraight"

# setting output related
output_prefix="/home/yang/data/aws_data/CIL_modular_data/benchmark_all/"
mkdir $output_prefix$expid
output_prefix=$output_prefix$expid"/"

# the town01
output_folder="/home/yang/data/aws_data/CIL_modular_data/_benchmarks_results/"$expid"_1,2,3,4,5,6,7,8,9,10,11,12,13,14_YangExp3cam_Town01/_images/"
ln -s $output_folder $output_prefix"Town01"
python eval_par.py -gpu "[4,5,6]" -expid $expid &


# RFS parked car
this_output=$output_prefix$"RFS_parked_car/"
mkdir $this_output
this_output=$this_output$"video"
/home/yang/data/aws_data/carla_rfs/CarlaUE4.sh -benchmark -fps=5 -carla-world-port=2600 &
export CARLA_VERSION="0.9.X"
sleep 25
python utils/eval_inenv_carla_090.py \
    --gpu 6 \
    --condition 2.0 \
    --expid $expid \
    --output_video_path $this_output \
    --starting_points "town03_intersections/positions_file_RFS_MAP.parked_car_attract.txt" \
    --parked_car "town03_intersections/positions_file_RFS_MAP.parking_v2.txt" \
    --townid "10" \
    --port 2600 &

#RFS shoulder
this_output=$output_prefix$"RFS_shoulder/"
mkdir $this_output
this_output=$this_output$"video"
/home/yang/data/aws_data/carla_rfs/CarlaUE4.sh -benchmark -fps=5 -carla-world-port=2700 &
export CARLA_VERSION="0.9.X"
sleep 25
python utils/eval_inenv_carla_090.py \
    --gpu 5 \
    --condition 2.0 \
    --expid $expid \
    --output_video_path $this_output \
    --starting_points "town03_intersections/positions_file_RFS_MAP.extra_explore_v3.txt" \
    --parked_car "" \
    --townid "10" \
    --port 2700 &


#ExpTown random
this_output=$output_prefix$"ExpTown_parked_car/"
mkdir $this_output
this_output=$this_output$"video"
/home/yang/data/aws_data/carla_095/CarlaUE4.sh Exp_Town -benchmark -fps=5 -carla-world-port=2800 &
export CARLA_VERSION="0.9.5"
sleep 25
python utils/eval_inenv_carla_090.py \
    --gpu 4 \
    --condition 2.0 \
    --expid $expid \
    --output_video_path $this_output \
    --starting_points "town03_intersections/positions_file_Exp_Town.parking_attract.txt" \
    --parked_car "town03_intersections/positions_file_Exp_Town.parking.txt" \
    --townid "11" \
    --port 2800 &