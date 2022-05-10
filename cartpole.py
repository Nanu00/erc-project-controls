import gym
from collections import deque

class Agent:
    def __init__(self, state, target, pos_gains = [1, 1, 1], angle_gains = [1, 1, 1]):
        self.cum_angle_error = 0
        self.cum_pos_error = 0

        self.angle_errors = deque([0, 0, 0, 0], 4)
        self.pos_errors = deque([0]*20, 20)

        self.update(state, target)

        self.angle_gains = angle_gains
        self.pos_gains = pos_gains

    def update(self, state, target):
        self.state = state
        self.target = target

        try:
            self.angle_error_delta = state[2] - target[2] - self.angle_error
        except:
            self.angle_error_delta = 0
        self.angle_error = state[2] - target[2]
        self.angle_errors.append(self.angle_error)
        self.angle_error_delta = sum(list(self.angle_errors)[int(len(self.angle_errors)/2):])/int(len(self.angle_errors)) - sum(list(self.angle_errors)[:int(len(self.angle_errors)/2)])/int(len(self.angle_errors))

        try:
            self.pos_error_delta = state[2] - target[2] - self.pos_error
        except:
            self.pos_error_delta = 0
        self.pos_error = state[2] - target[2]
        self.pos_errors.append(self.pos_error)
        self.pos_error_delta = sum(list(self.pos_errors)[int(len(self.pos_errors)/2):])/int(len(self.pos_errors)) - sum(list(self.pos_errors)[:int(len(self.pos_errors)/2)])/int(len(self.pos_errors))

        self.cum_angle_error += self.angle_error
        self.cum_pos_error += self.pos_error

    def get_action(self):
        output_angle = self.angle_gains[0] * self.angle_error + self.angle_gains[1] * self.cum_angle_error + self.angle_gains[2] * self.angle_error_delta
        output_pos = self.pos_gains[0] * self.pos_error + self.pos_gains[1] * self.cum_pos_error + self.pos_gains[2] * self.pos_error_delta

        output = output_angle + output_pos

        # print(f"{output_angle}, {output_pos}")

        if output > 0:
            return 1
        else:
            return 0

env = gym.make('CartPole-v1')

state = env.reset()

target = state
target[0] = 0
target[2] = 0

a = Agent(state, target, [0.2, 0.2, 0.2], [0.2, 0.1, 10])
done = False

while not done:
    env.render()
    state, _, done, _ = env.step(a.get_action())
    a.update(state, a.target)

env.close()
