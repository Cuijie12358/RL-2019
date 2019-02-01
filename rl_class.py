import numpy as np
########################################################
#
# Class for Multi Armed Bandit Problem:
# class Multi10BanditEnv:
#       mu_arms scale: Contain the actual values of each arm(10 arms)
# class Agent:
#       step:   Contain the current step,
#       reward: all the rewards,
#       A:      times of choosing each arm,
#       Q_a:    action-value of choosing ath arm
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

########################### Test code
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

#
# loop = 100
# test_a = Agent(0.1)
# step = 200
# average_reward=np.zeros(step)
# optimal_action=np.zeros(step)
# for i in range(loop):
#     average_reward_1 = []
#     optimal_action_1 = []
#     test_a.agent_reset()
#     while test_a.step < step:
#         test_a.update_state()
#         test_a.take_action()
#         average_reward_1=np.append(average_reward_1, test_a.get_average_reward())
#         optimal_action_1=np.append(optimal_action_1, test_a.get_optimal_action())
#     print(test_a.A)
#     average_reward += np.array(average_reward_1)
#     optimal_action += np.array(optimal_action_1)
# average_reward /= loop
# optimal_action /= loop


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
Direction = np.array(['UP','RIGHT','DOWN','LEFT'])

class GridWorld_DP:
    Vs = np.zeros(16)
    Policy = 0.25 * np.ones((16,4))
    reward = -1
    theta = 0.1

    def __init__(self):
        return None

    def get_pis(self):
    	return self.Policy

    def get_Vs(self):
        return self.Vs

    def set_vs(self,vs):
        self.Vs=vs

    # up, right, down, left. The same as UCL did.
    def find_Vs4(self,i):
        s1, s2, s3, s4 = i-4,i+1,i+4,i-1
        if s1 < 0:
            s1 = i
        if s2 % 4 == 0:
            s2 = i
        if s3 > 15:
            s3 = i
        if (s4 != 0) and (s4%4==3) :
            s4 = i
        return np.array([self.Vs[s1],self.Vs[s2],self.Vs[s3],self.Vs[s4]])

    def policy_evaluation(self):
        next_Vs = np.zeros(16)
        for i in range(1,15):
            Vs4 = self.find_Vs4(i)
            next_Vs[i] = np.round(np.sum(self.Policy[i]*1 * (self.reward + Vs4)),2)
        return next_Vs

    def policy_improvement(self):
        nextV_ifpi = np.zeros(4)
        for i in range(1,15):
            Vs4 = self.find_Vs4(i)
            nextV_ifpi[:] = 1 * (self.reward + Vs4)-self.Vs[i]
            self.Policy[i, nextV_ifpi != np.max(nextV_ifpi)] = 0
            self.Policy[i]=self.Policy[i]/np.sum(self.Policy[i])

    def value_iteration(self):
        next_Vs = np.zeros(16)
        self.policy_improvement()
        for i in range(1,15):
            Vs4 = self.find_Vs4(i)
            next_Vs[i] = np.round(np.sum(self.Policy[i]*1 * (self.reward + Vs4)),2)
        return next_Vs


# gridworld = GridWorld_DP()
# delta = 999
# threshold = 0.01
# K=0
# while delta > threshold:
#     delta = 0
#     V = gridworld.get_Vs()
#     print("K=",K)
#     print(np.reshape(V,(4,4)))
#     next_v = gridworld.policy_evaluation()
#     K+=1
#     gridworld.set_vs(next_v)
#     delta = max([delta,np.max(np.abs(next_v-V))])
# print("The evaluation converges at last!")
# gridworld = GridWorld_DP()
# delta = 999
# threshold = 0.01
# K=0
# old_action = gridworld.get_pis()
# new_action = np.array(old_action)+1
# while np.sum(old_action - new_action):
#     # policy evaluation
#     while delta > threshold:
#         delta = 0
#         V = gridworld.get_Vs()
#         # print("K=",K)
#         # print(np.reshape(V,(4,4)))
#         next_v = gridworld.policy_evaluation()
#         gridworld.policy_improvement()
#
#         K+=1
#         gridworld.set_vs(next_v)
#         delta = max([delta,np.max(np.abs(next_v-V))])
#
#     # policy improvement
#
#     old_action = gridworld.get_pis()
#     gridworld.policy_improvement()
#     new_action = gridworld.get_pis()
#
#     # focus on the state 9
#     print("DIRECTION IN STATE 9:",Direction[gridworld.Policy[9]!=0])
#
#
# print(np.reshape(gridworld.get_Vs(),(4,4)))
# print("The evaluation converges at last!")

gridworld = GridWorld_DP()
delta = 999
threshold = 0.1
K=0
while delta > threshold:
    delta = 0
    V = gridworld.get_Vs()
    print("K=",K)
    print(np.reshape(V,(4,4)))
    next_v = gridworld.value_iteration()
    K+=1
    gridworld.set_vs(next_v)
    delta = max([delta,np.max(np.abs(next_v-V))])
print("The evaluation converges at last!")

