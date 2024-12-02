import random
import time
import numpy as np
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

# 引用上级目录
import sys
sys.path.append("..")
import grid_env

class Q_learning():
    def __init__(self,alpha,env = grid_env.GridEnv):
        self.gamma = 0.9  # discount rate
        self.alpha = alpha  #learning rate
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(
            self.env.reward_list), self.env.reward_list  # [-10,-10,0,1]  reward list
        self.state_value = np.zeros(shape=self.state_space_size)  # 一维列表
        print("self.state_value:", self.state_value)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))  # 二维： state数 x action数
        self.mean_policy = np.ones(     #self.mean_policy shape: (25, 5)
            shape=(self.state_space_size, self.action_space_size)) / self.action_space_size  # 平均策略，即取每个动作的概率均等
        self.policy = self.mean_policy.copy()
        # self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象

        print("action_space_size: {} state_space_size：{}".format(self.action_space_size, self.state_space_size))
        print("state_value.shape:{} , qvalue.shape:{} , mean_policy.shape:{}".format(self.state_value.shape,
                                                                                     self.qvalue.shape,
                                                                                     self.mean_policy.shape))

        print('----------------------------------------------------------------')

    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)

    '''
        Learn an optimal policy that can lead the agent to the target state from an initial state s0.
    '''

    def q_learning(self, initial_location, epsilon=1.0, specific_direction=None):
        total_rewards = []
        episode_lengths = []
        err=[]
        initial_state = self.env.pos2state(initial_location)
        print("initial_state:", initial_state)
        # TODO: 行为策略 误差值绘制
        for episode_num in range(1):  # episode_num
            self.env.reset()
            total_reward = 0
            episode_length = 0
            done = False
            print("episode_num:", episode_num)
            state = initial_state
            # while not done:
            cnt=10000
            while cnt>0:
                # Choose action using epsilon-greedy policy 
                if specific_direction is None:
                    if np.random.rand() < epsilon:
                        action = np.random.choice(self.action_space_size)  # Explore: random action
                    else:
                        action = random_matrix[state]  # Exploit: 选择特定
                else:
                    if np.random.rand() < epsilon:
                        action = np.random.choice(self.action_space_size)  # Explore: random action
                    else:
                        action = specific_direction  # Exploit: 选择特定的方向

                # Take action and observe reward and next state
                _, reward, done, _, _ = self.env.step(action)
                next_state = self.env.pos2state(self.env.agent_location)

                # Update Q-value
                best_next_action = np.argmax(self.qvalue[next_state])
                td_target = reward + self.gamma * self.qvalue[next_state][best_next_action]
                td_error = self.qvalue[state][action] - td_target
                self.qvalue[state][action] -= self.alpha * td_error

                # Update policy (optional, since Q-learning is off-policy)
                qvalue_star = self.qvalue[state].max()
                self.state_value[state]=qvalue_star
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        # self.policy[state, a] = 1 - epsilon + (epsilon / self.action_space_size)
                        self.policy[state, a] = 1 
                    else:
                        # self.policy[state, a] = epsilon / self.action_space_size
                        self.policy[state, a] = 0

                # Update state
                state = next_state
                total_reward += reward
                episode_length += 1
                cnt-=1
                err.append(np.mean(np.abs(self.state_value-optimal_policy)))
            # total_rewards.append(total_reward)
            # episode_lengths.append(episode_length)

        return err
if __name__ =="__main__":
    gird_world = grid_env.GridEnv(size=5, target=[2, 3],
                                  forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [3, 3], [1, 4]],
                                  render_mode='')

    solver = Q_learning(alpha =0.1, env = gird_world)
    start_time = time.time()

    initial_location = [4,0]
    optimal_policy=np.array([ 5.8, 5.6, 6.2, 6.5, 5.8, 6.5, 7.2, 8.0, 7.2, 6.5, 7.2, 8.0, 10.0, 8.0, 7.2, 8.0, 10.0, 10.0,10.0, 8.0, 7.2, 9.0, 10.0, 9.0, 8.1])
    random_matrix = np.random.randint(low=0, high=5, size= 25)
    print(random_matrix)
    err= solver.q_learning(initial_location = initial_location,epsilon=0.1,specific_direction=None)

    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:",cost_time)
    print(len(gird_world.render_.trajectory))

    # solver.show_policy()  # solver.env.render()
    # solver.show_state_value(solver.state_value, y_offset=0.25)
    solver.env.render_.draw_episode()

    gird_world.render()
  
    print("--------------------")
    print("Plot")
    # 绘制第一个图表
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(err) + 1), err,   # 空心，设置填充色为透明
             markeredgecolor='blue',  # 边框颜色为蓝色
             markersize=10,
             linestyle='-', color='blue',label = "error")
    plt.ylim([0,8])
    plt.xlabel('Episode index', fontsize=12)
    plt.ylabel('error', fontsize=12)

    # 添加图例
    plt.legend()
    # 显示图表
    plt.show()
