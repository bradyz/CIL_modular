import copy, random, time, os, sys, inspect
def get_file_real_path():
    abspath = os.path.abspath(inspect.getfile(inspect.currentframe()))
    return os.path.realpath(abspath)

sys.path.append(os.path.join(os.path.dirname(get_file_real_path()), "../utils"))
import common_util

__CARLA_VERSION__ = os.getenv('CARLA_VERSION', '0.8.X')
common_util.add_carla_egg_to_path(__CARLA_VERSION__)
import carla
from carla import VehicleControl as VehicleControl

class Noiser(object):
    # define frequency into noise events per minute
    # define the amount_of_time of setting the noise

    # NOISER CARLA CONFIGURATION
    # frequency=15, intensity = 5 ,min_noise_time_amount = 0.5

    def __init__(self, noise_type, frequency=45, intensity=0.004, min_noise_time_amount=0.025,
                 no_noise_decay_stage=False,
                 use_tick=False,
                 time_amount_multiplier=1.0,
                 noise_std = 1.0,
                 no_time_offset=False):
        # specifications from outside
        self.noise_type = noise_type
        self.frequency = frequency
        self.intensity = intensity
        self.min_noise_time_amount = min_noise_time_amount
        self.no_noise_decay_stage = no_noise_decay_stage
        self.use_tick = use_tick
        self.time_amount_multiplier = time_amount_multiplier
        self.noise_std = noise_std
        self.no_time_offset = no_time_offset

        # FSM state variables
        if use_tick:
            self.noise_start_time = 0
            self.noise_end_time = 1
            self.second_counter = 0
            self.tick = 0.0
        else:
            self.noise_start_time = time.time()
            self.noise_end_time = time.time() + 1
            self.second_counter = time.time()
        self.spike_decay_stage = False
        self.spike_rise_stage = False

        # noise parameter variables
        self.noise_time_amount = min_noise_time_amount + float(random.randint(50, 200) / 100.0) * self.time_amount_multiplier
        self.noise_sign = 1.0

    def return_time(self):
        if self.use_tick:
            return self.tick
        else:
            return time.time()

    def randomize_noise_sign(self):
        if self.noise_type == 'Spike':  # spike noise there are no variations on current noise over time
            self.noise_sign = float(random.randint(0, 1) * 2 - 1)
            if self.use_tick:
                self.this_intensity = (random.random()*self.noise_std + 0.5) * self.intensity
            else:
                self.this_intensity = self.intensity * random.random() * 5 + 2.5

    def get_noise(self):
        assert(self.noise_type == 'Spike')
        # first compute for positive noise
        if self.spike_rise_stage:
            time_offset = self.return_time() - self.noise_start_time
        elif self.spike_decay_stage:
            time_offset = (self.noise_end_time - self.noise_start_time) - (self.return_time() - self.noise_end_time)
        else:
            raise ValueError()

        if self.no_time_offset:
            factor = 1.0
        else:
            factor = time_offset
        noise = 0.001 + factor * 0.03 * self.this_intensity
        noise = min(noise, 0.55) * self.noise_sign

        return noise

    def is_time_for_noise(self):
        # This implements a FSM, where there are 3 states: no noise, spike rise stage, spike decay stage
        # At state no noise, if a second passed, with self.frequency/60, we will enter the spike rise stage
        # all state variables:
        # start_time, end_time, second_counter, spike_rise_stage, spike_decay_stage
        # noise parameters: noise_time_amount, noise_sign

        if not self.spike_rise_stage and not self.spike_decay_stage:
            # state no noise
            if self.return_time() - self.second_counter >= 5.0:
                self.second_counter = self.return_time()
                if random.randint(0, 60) < self.frequency:
                    self.spike_rise_stage = True
                    self.randomize_noise_sign()
                    self.noise_time_amount = self.min_noise_time_amount + random.randint(50, 200) / 100.0 * self.time_amount_multiplier
                    self.noise_start_time = self.return_time()
                    return True
                else:
                    return False
            else:
                return False

        elif self.spike_rise_stage:
            assert(self.spike_decay_stage == False)
            if self.return_time() - self.noise_start_time >= self.noise_time_amount:
                self.spike_rise_stage = False
                if self.no_noise_decay_stage:
                    self.spike_decay_stage = False
                    self.second_counter = self.return_time()
                    return False
                else:
                    self.spike_decay_stage = True
                self.noise_end_time = self.return_time()
            return True

        elif self.spike_decay_stage:
            assert(self.spike_rise_stage == False)
            if self.return_time() - self.noise_end_time > self.noise_time_amount:
                self.spike_decay_stage = False
                self.second_counter = self.return_time()
                return False
            else:
                return True

    def compute_noise(self, action, speed_kmh=20):
        if self.use_tick:
            self.tick += 0.2

        if self.noise_type == 'None':
            return action

        if self.noise_type == 'Spike':
            if self.is_time_for_noise():
                minmax = lambda x: max(x, min(x, 1.0), -1.0)
                steer_noisy = minmax(action.steer + self.get_noise() * (30 / (1.5 * speed_kmh + 5)))

                if __CARLA_VERSION__.startswith("0.9"):
                    noisy_action = VehicleControl(action.throttle, action.steer, action.brake, action.hand_brake, action.reverse)
                else:
                    noisy_action = copy.deepcopy(action)
                noisy_action.steer = steer_noisy
                return noisy_action

            else:
                return action

        raise ValueError("invalid noisy type: " + self.noise_type)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Noise testing
    noise_input = []
    human_input = []
    noiser = Noiser('Spike')
    for i in range(500):
        human_action = Control()
        human_action.steer = 0.0
        human_action.gas = 0.0
        human_action.brake = 0.0
        human_action.hand_brake = 0.0
        human_action.reverse = 0.0

        human_input.append(human_action.steer)

        noisy_action, _, _ = noiser.compute_noise(human_action)
        time.sleep(0.01)
        noise_input.append(noisy_action.steer)

    plt.plot(list(range(500)), human_input, 'g', list(range(500)), noise_input, 'r')
    plt.show()
