import numpy as np

class Env:

    def __init__(self):
        self.state = np.array([0,0,0,0,0,0,0,0,0])
        self.c = -1

    def reset(self):
        self.state = np.array([0,0,0,0,0,0,0,0,0])
        self.c = -1

    def start(self):
        c = np.random.choice(6)
        if c == 0:
            row = np.random.choice(3)
            self.state[row*3] = 1
            self.state[row*3 + 1] = 1
            self.c = row*3+2
        elif c == 1:
            row = np.random.choice(3)
            self.state[row*3 + 2] = 1
            self.state[row*3 + 1] = 1
            self.c = row*3
        elif c == 2:
            column = np.random.choice(3)
            self.state[column] = 1
            self.state[column + 3] = 1
            self.c = column + 6
        elif c == 3:
            column = np.random.choice(3)
            self.state[column + 6] = 1
            self.state[column + 3] = 1
            self.c = column
        elif c == 4:
            self.state[6] = 1
            self.state[4] = 1
            self.c = 2
        elif c == 5:
            self.state[2] = 1
            self.state[4] = 1
            self.c = 6
        
        return self.state

    def place(self, index):
        reward = 1 if index == self.c else 0
        self.state[index] = 1
        return self.state, reward


def to_index(state):
    num = 0
    for i in range(8):
        num |= state[i]
        num = num << 1

    return num | state[8]

lr = 0.1
epsilon = 1
epsilon_discount = 0.999
q_table = np.zeros((512, 9))
env = Env()
for i in range(1000):
    env.reset()
    s = env.start()
    
    s_i = to_index(s)
    if(np.random.rand() < epsilon):
        a = np.random.choice(9)
    else:
        a = np.argmax(q_table[s_i])

    s, r = env.place(a)
    
    q_table[s_i, a] = (1-lr) * q_table[s_i, a] + lr * r 
    epsilon *= epsilon_discount
    print(i, epsilon)


for i in range(5):
    env.reset()
    s = env.start().copy()
    np.reshape(s, (-1, 3))
    
    s_i = to_index(s)
    a = np.argmax(q_table[s_i])

    s2, r = env.place(a)
    print("==== Finish the line ===")
    print("Start: ")
    print(np.reshape(s, (-1, 3)))
    print("After:")
    print(np.reshape(s2, (-1, 3)))