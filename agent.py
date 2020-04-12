"""
A Q-learning and UCB Agent that will be used to explore the environment. The agent will
Take an action (Select a channel to try and send a message on) and be rewarded if
successfully picks an unused channel

Author: Brandon Dominique
Email: dominiquebdmnq@aol.com
Date:8/2/19
"""
import numpy as np
import math
from environment import Environment

class Agent:
    def QLearning(epsilon, min_eps, episodes):
        """
        the observation is an array of 5 values - the amount of energy on each
        of the 5 channels. It's the job of the Agent to look at the energy levels
        and select from them the channel that it can send data on (it will always
        be the one with the lowest amount of energy). Ideally all of the channels
        have the same PSK modulation number and different SNRs. Not all 5 channels will
        be active at once, so its up to the Agent to select the channel that is below the
        threshold value (which indicates that no one's using it at the moment).
        """
        
        env = Environment(-10,10)

        accuracy_list = []
        episode_times = []
        
        channel_index = [0,1,2,3]
        
        obs_space = []

        q = np.zeros((len(channel_index)))
        
        """
        q is a 2*4 table of zeros. after every episode the table will be updated depending on the
        result of the transmission. the first row of 4 represents the Q-value of each channel
        when it is picked and its below the threshold value, and the second row represents  the
        Q-value for each channel when its is picked and it's above the threshold value. After
        a large amount of episodes the Q-vlaues of each channel in the
        first row should be >>> the values in the second (a channel with a value higher than
        the threshold shouldn't be picked, because someone's transmitting on that channel.)
        
        """
        reduction = (epsilon - min_eps)/episodes
        
        #call the array of 4 values at 4 predetermined points

        #based on the channel_number given, check if the value at that array index
        #is over the threshold value for each episode.

        for i in range(episodes):
            obs_space = env.reset()
            reward = 0
            print("BERs for Episode ",i+1,": ",obs_space)
                 
            if np.random.random() < 1 - epsilon:
                action = np.argmax(q[0])
            else:
                action = np.random.randint(0,4)

            
            if obs_space[action] != env.ber1:
                print("Correct Channel used! (Channel ", action+1,")")
                reward = 1
                q[action] += reward
            else:
                print("Incorrect Channel used! (Channel ", action+1,")")
                reward = -1
                q[action] += reward


            if epsilon > min_eps:
                epsilon -= reduction
                
            accuracy = np.abs(np.sum(q))/(i+1)

            accuracy_list.append(accuracy)
            episode_times.append(i+1)

            if (i+1) % 10 == 0:
                print("Episode ",(i+1))
                print("Total Reward: ", np.sum(q))
                print("Total Possible Reward: ", (i+1))
                print(np.abs(np.sum(q))/(i+1), "% Accuracy")
        return q, np.sum(q), episode_times

    
    def UCB(episodes):
        """
        Upper Confidence Bound Algorithm that can be compared to QLearning() above.
        The basic idea of the Algorithm is to try every action available to it
        at least once and then start choosing options based on a formaula between the
        reward it gets for an option and a logarithmic square root function. I'm not yet
        sure if this means that my agent must try each option once (select each of the 5
        channels) or try each option with success and failure once (select each of the 5
        options in a correct scenario and then again in an incorrect scenario).
        """

        #initialize a counts array (count the number of times each action is called)
        #and values (the value of each action). values is a 2*4 array like the Q table
        #in QLearning() so that we can record the number of correct and incorrect decisions.
        counts = [0,0,0,0]
        values = [[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0]]
        obs_space = []
        env = environment(-10,10)
        accuracy_list = []
        episode_times = []
        
        #initalize the ucb table to keep track of the value for each action
        ucb_values = [0.0, 0.0, 0.0, 0.0]


        for i in range(episodes):

            """
            Make sure that each action has been performed at least once. If count of an arm is == 0 ,
            take that action. otherwise, determine the highest UCB value between each action and take that one.
            after that, update the values and counts tables and loop through again.
            """
            
            obs_space = env.reset()
            print("BERs for Episode ",i+1,": ",obs_space)
            number_of_actions = len(counts)
            action_to_take = 0

            if np.all(counts) == False: # If there is an index of counts that is 0:
                for action in range(number_of_actions):
                    if counts[action] == 0:
                        action_to_take = action
                        break
            else: # if every index of counts is non-zero (Every option has been tried once):
                total_count = sum(counts)
                
                for action in range(number_of_actions):
                    bonus = math.sqrt((2*math.log(total_count)) / float(counts[action]))
                    ucb_values[action] = values[0][action] + bonus
                action_to_take = np.argmax(ucb_values)

            counts[action_to_take] = counts[action_to_take] + 1
            
            
            #check to see whether the selected channel has a SnR higher than the threshold value
            

            reward = 0
        
            if obs_space[action_to_take] < env.threshold_value: #threshold_value still has to be determined
                print("Correct Channel used! (Channel ", action_to_take+1,")")
                reward = 1
                values[0][action_to_take] += reward
            else:
                print("Incorrect Channel used! (Channel ", action_to_take+1,")")
                reward = -1
                values[1][action_to_take] += reward
            """
            n = counts[action_to_take]
            value = values[action_to_take]
            new_value = ((n*1)/float(n))*value + (1 / float(n)) * reward
            values[action_to_take] = new_value
            """
        
            accuracy = np.sum(values[0])/(i+1)

            accuracy_list.append(accuracy)
            episode_times.append(i+1)
            

            
            
            if (i+1) % 10 == 0:
                print("Episode ",(i+1))
                print("Number of times the correct channel was picked: ", np.sum(values[0]))
                print("Number of times the incorrect channel was picked: ", np.sum(np.abs(values[1])))
                print(np.sum(values[0])/(i+1), "% Accuracy")
        
        return values,accuracy_list,episode_times    
            

            

            


            
                
            
