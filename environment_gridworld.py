import numpy as np

class environment():
    def __init__(self):
        self.reset()

    def reset(self):
        self.position = [0,0]
        self.goal = [2,2]
        self.steps = 0

    def take_action(self, action):
        waypoint = [int(action%3), int(action/3)]
        
        self.position[0] += np.sign(waypoint[0]-self.position[0])*0.5
        self.position[1] += np.sign(waypoint[1]-self.position[1])*0.5

        reward = self.getReward()
        done = self.isEnd()
        self.steps += 1

        return self.position, reward, done


    def getReward(self):
        if (self.position[0]>self.goal[0]-0.5 and self.position[0]<self.goal[0]+0.5) and (self.position[1]>self.goal[1]-0.5 and self.position[1]<self.goal[1]+0.5):
            return 10
        else:
            return -1
    
    def isEnd(self):
        if (self.position[0]>self.goal[0]-0.5 and self.position[0]<self.goal[0]+0.5) and (self.position[1]>self.goal[1]-0.5 and self.position[1]<self.goal[1]+0.5):
            return True
        elif self.steps >= 100:
            return True
        else:
            return False


def test_environment():
    env = environment()
    n = 0

    while n<=5:

        action = int(input())
        state, reward, done = env.take_action(action)
        
        print(f"state = {state}")
        print(f"reward =  {reward}")
        print(f"done = {done}")
        #print(f'Position = {env.position}')
        #print(f'State = {env.state}')

        if env.isEnd():
            env.reset()
            n+=1
            print(f"n = {n}")


#test_environment()

        

    

