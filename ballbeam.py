import gym
import ballbeam_gym
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, gains = [1, 1, 1]):
        self.gains = gains
        self.cum_error = 0
        self.pos_data = {}
        self.vel_data = {}
        self.step = 0

    def get_action(self, env):
        error = (env.bb.x - env.setpoint)
        self.cum_error += error
        angle_step = self.gains[0]*error + self.gains[1]*self.cum_error + self.gains[2]*(env.bb.v)

        self.pos_data[self.step] = env.bb.x
        self.vel_data[self.step] = env.bb.v

        self.step += 1

        return angle_step

    def graph(self):
        plt.plot(self.pos_data.keys(), self.pos_data.values())
        plt.plot(self.vel_data.keys(), self.vel_data.values())
        plt.show()

kwargs = {
    'timestep': 0.05,
    'beam_length': 1.0,
    'setpoint': 0.0,
    'max_angle': 0.5,
    'init_velocity': 0.5,
    'max_timesteps': 100,
    'action_mode': 'continuous'
}
env = gym.make('BallBeamSetpoint-v0', **kwargs)

a = Agent([10, 0, 1.2])

env.reset()
done = False

for _ in range(100):
    env.render()
    _, _, done, _ = env.step(a.get_action(env))

print(f"Ball position: {env.bb.x}")
a.graph()

env.close()
