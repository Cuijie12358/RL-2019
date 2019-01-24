import numpy as np


class Multi10BanditEnv:
    """10-Armed Bandit with their reward distribution"""
    mu_arms = [0.2, -0.6, 1.5, 0.6, 1.2, -1.6, -0.2, -1.2, 0.7, -0.7]
    scale = 3

    def __init__(self):
        return None

    def get_reward(self, action):
        reward = self.mu_arms[action - 1] + self.scale * np.random.rand()
        return reward

    def get_optimal_arm(self):
        return np.argmax(self.mu_arms) + 1


class Agent:
    """Setting parameters for an agent"""
    Q_a = 10 * np.ones(10)
    A = np.zeros(10)
    step = 0
    reward = 0
    optima = 0

    def __init__(self, epsilon=0.1):
        self.env = Multi10BanditEnv()
        self.epsilon = epsilon
        self.action = np.random.randint(10) + 1
        self.optima = self.env.get_optimal_arm()

    def update_state(self):
        self.step += 1
        self.reward = self.env.get_reward(self.action)
        sum_r = self.A[self.action - 1] * self.Q_a[self.action - 1] + self.reward
        self.A[self.action - 1] += 1
        self.Q_a[self.action - 1] = sum_r / self.A[self.action - 1]

    def take_action(self):
        if np.random.rand() < self.epsilon:
            self.action = np.random.randint(10) + 1
        else:
            self.action = np.argmax(self.Q_a) + 1

    def get_average_reward(self):
        return np.dot(self.Q_a, self.A) / self.step

    def get_optimal_action(self):
        return self.A[self.optima - 1] / self.step

    def agent_reset(self):
        self.Q_a = 10 * np.ones(10)
        self.A = np.zeros(10)
        self.step = 0
        self.reward = 0
        self.action = np.random.randint(10) + 1






# test_a = Agent(0.1)
# step = 15
# average_reward = []
# optimal_action = []
# while test_a.step < step:
#     test_a.update_state()
#     test_a.take_action()
#     average_reward=np.append(average_reward, test_a.get_average_reward())
#     optimal_action=np.append(optimal_action, test_a.get_optimal_action())
#     print(test_a.Q_a)


loop = 100
test_a = Agent(0.1)
step = 200
average_reward=np.zeros(step)
optimal_action=np.zeros(step)
for i in range(loop):
    average_reward_1 = []
    optimal_action_1 = []
    test_a.agent_reset()
    while test_a.step < step:
        test_a.update_state()
        test_a.take_action()
        average_reward_1=np.append(average_reward_1, test_a.get_average_reward())
        optimal_action_1=np.append(optimal_action_1, test_a.get_optimal_action())
    print(test_a.A)
    average_reward += np.array(average_reward_1)
    optimal_action += np.array(optimal_action_1)
average_reward /= loop
optimal_action /= loop

