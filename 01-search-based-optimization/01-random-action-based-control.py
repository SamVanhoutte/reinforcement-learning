import gymimport seaborn as snsenv = gym.make('CartPole-v0')results = list()for i_episode in range(1000):    observation = env.reset()    for t in range(100):        env.render()        #print(observation)        action = env.action_space.sample()        observation, reward, done, info = env.step(action)        if done:            results.append(t+1)            #print("Episode finished after {} timesteps".format(t+1))            breakenv.close()sns.distplot(results)print("Average number of steps during experiment {}".format(sum(results) / len(results)))