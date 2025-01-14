from subprocess import Popen
from time import sleep
import math, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluation in parallel')
    parser.add_argument('-gpu', '--gpu_perception_agent', default="[1,2,3]", help="which gpu to use")
    parser.add_argument('-expid', '--expid', default="mm45_v4_SqnoiseShoulder_exptownv3_notown0102_mergefollowstraight", help="expid")
    parser.add_argument('-townid', '--townid', default='["Town01"]', help="which town to run on")
    args = parser.parse_args()

    gpus_agent = eval(args.gpu_perception_agent)
    gpus_carla = [0]
    gpus_perception = eval(args.gpu_perception_agent)
    num_perception = 1
    exp_id = args.expid
    weather_batch_size = 14 #7
    test_name = "YangExp3cam"
    town_list = eval(args.townid) #["Town01", "Town02"]
    #test_name = "YangExp3camFov90"
    #test_name = "YangExp3camGTA"
    # TODO make a new test setting to mimic the camera locations
    # num par = 14/3 * 2


    processes = []
    ithread = 0

    for town in town_list:
        next_weather = 1
        for _ in range(int(math.ceil(14.0 / weather_batch_size))):
            weather_id = ""
            for i in range(weather_batch_size):
                if next_weather > 14:
                    break
                weather_id += str(next_weather) + ","
                next_weather += 1
            weather_id = weather_id[:-1]

            percep = ""
            for i in range(num_perception):
                id = ithread*num_perception+i
                percep += str(gpus_perception[id % len(gpus_perception)]) + ","
            percep = percep[:-1]


            cmd = ["./eval_one.sh",
                   str(gpus_agent[ithread % len(gpus_agent)]),
                   str(gpus_carla[ithread % len(gpus_carla)]),
                   percep,
                   weather_id,
                   exp_id,
                   town,
                   test_name]

            # TODO: call the eval once code
            print(cmd)
            p=Popen(cmd)
            processes.append(p)
            sleep(10)

            ithread += 1

    for p in processes:
        p.wait()
