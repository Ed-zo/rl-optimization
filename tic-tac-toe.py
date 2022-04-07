import numpy as np

class Env:

    def __init__(self, against_AI = True):
        self.state = np.array([0,0,0,0,0,0,0,0,0])
        self.ai = against_AI

    def reset(self):
        self.state = np.array([0,0,0,0,0,0,0,0,0])
        self.bot_step()

        return self.state

    def check_win(self):
        #riadky
        for i in range(3):
            if self.state[i*3] != 0 and self.state[i*3] == self.state[i*3 + 1] and self.state[i*3] == self.state[i*3 + 2]:
                return self.state[i*3]

        #stlpce
        for i in range(3):
            if self.state[i] != 0 and self.state[i] == self.state[i + 3] and self.state[i] == self.state[i + 6]:
                return self.state[i]

        if self.state[i] != 0 and self.state[i] == self.state[i + 3] and self.state[i] == self.state[i + 6]:
            return self.state[i]

        #diagonala
        if self.state[0] != 0 and self.state[0] == self.state[4] and self.state[0] == self.state[8]:
            return self.state[0]

        if self.state[2] != 0 and self.state[2] == self.state[4] and self.state[2] == self.state[6]:
            return self.state[2]

        return 0

    def print(self):
        print(np.reshape(self.state, (-1, 3)))

    def bot_step(self):
        if self.ai:
            #random choice
            zeros = np.where(self.state == 0)[0]
            bot_i = zeros[np.random.choice(zeros.size)]

            self.state[bot_i] = 1
        else:
            print('Your position: ')
            player_i = int(input())
            while self.state[player_i] != 0:
                player_i = int(input())

            self.state[player_i] = 1

    def step(self, index):
        if self.state[index] == 0:
            self.state[index] = 2
        
        if(not self.ai):
            env.print()

        win = self.check_win()
        full = np.all(self.state)
        if not full and not win:
            self.bot_step()
            win = self.check_win()
        full = np.all(self.state)

        return self.state, 1 if win == 2 else 0, win != 0 or full

def to_index(state):
    num = 0
    for i in range(9):
        if state[i] == 1:
            num |= 1
        if state[i] == 2:
            num |= 2
        num = num << 2

    return num >> 2


lr = 0.1
gamma = 0.98
epsilon = 1
epsilon_discount = 0.99995
env = Env()

training = False
if training:
    with open('q-table.npy', 'rb') as f:
        q_table = np.load(f)
    # q_table = np.zeros((262144, 9))
    s, s_next = None, None
    for i in range(100000):
        s = env.reset()
        
        r_sum = 0
        terminal = False
        while not terminal:
            s_i = to_index(s)
            if(np.random.rand() < epsilon):
                zeros = np.where(s == 0)[0]
                a = zeros[np.random.choice(zeros.size)]
            else:
                a = np.argmax(q_table[s_i])

            s_next, r, terminal = env.step(a)
            r_sum += r
            q_table[s_i, a] = q_table[s_i, a] + lr * (r + gamma * np.max(q_table[to_index(s_next)]) - q_table[s_i, a])
            s = s_next
    
        epsilon = max(0.01, epsilon * epsilon_discount)
        print(i, epsilon, r)
    
    with open('q-table.npy', 'wb') as f:
        np.save(f, q_table)
else:
    with open('q-table.npy', 'rb') as f:
        q_table = np.load(f)

#TESTING
env = Env(False)
s = env.reset()
terminal = False
while not terminal:
    s_i = to_index(s)
    a = np.argmax(q_table[s_i])
    print(np.reshape(q_table[s_i], (-1, 3)))
    s, r, terminal = env.step(a)

env.print()
print("Wins player: ", env.check_win())