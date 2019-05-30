import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output

import random

class board:
    """simple implementation of 2048 puzzle"""
    
    def __init__(self, tile = None, max_number=15):
        self.tile = tile if tile is not None else [0] * 16
        self.max_num = max_number
    
    def __str__(self):
        state = '+' + '-' * 24 + '+\n'
        for row in [self.tile[r:r + 4] for r in range(0, 16, 4)]:
            state += ('|' + ''.join('{0:6d}'.format((1 << t) & -2) for t in row) + '|\n')
        state += '+' + '-' * 24 + '+'
        return state
    
    def mirror(self):
        return board([self.tile[r + i] for r in range(0, 16, 4) for i in reversed(range(4))])
    
    def transpose(self):
        return board([self.tile[r + i] for i in range(4) for r in range(0, 16, 4)])
    
    def rotate(self):
        return board([self.tile[4*(3-(i%4)) + (i//4)] for i in range(16)])
    
    def left(self):
        move, score = [], 0
        for row in [self.tile[r:r+4] for r in range(0, 16, 4)]:
            row, buf = [], [t for t in row if t]
            while buf:
                if len(buf) >= 2 and buf[0] is buf[1]:
                    buf = buf[1:]
                    buf[0] += 1
                    score += 1 << buf[0]
                row += [buf[0]]
                buf = buf[1:]
            move += row + [0] * (4 - len(row))
        return board(move), score if move != self.tile else -1
    
    def right(self):
        move, score = self.mirror().left()
        return move.mirror(), score
    
    def up(self):
        move, score = self.transpose().left()
        return move.transpose(), score
    
    def down(self):
        move, score = self.transpose().right()
        return move.transpose(), score
    
    def popup(self):
        tile = self.tile[:]
        empty = [i for i, t in enumerate(tile) if not t]
        tile[random.choice(empty)] = random.choice([1] * 9 + [2])
        return board(tile)
    
    def allpopup(self):
        tile = self.tile[:]
        empty = [i for i, t in enumerate(tile) if not t]
        boards = []
        for i in empty:
            for n in [1] * 9 + [2]:
                tmp = tile.copy()
                tmp[i] = n
                boards.append(board(tmp))
        return boards
    
    def end(self):
        tile = self.tile[:]
        empty = [i for i, t in enumerate(tile) if not t]
        
        count_max_num = np.count_nonzero(self.max_num == np.array(tile))
        return len(empty) == 0 or count_max_num > 0
    
def find_isomorphic_pattern(pattern):
    a = board(list(range(16)))

    isomorphic_pattern = []
    for i in range(8):
        if (i >= 4):
            b = board( a.mirror().tile )
        else:
            b = board( a.tile )
        for _ in range(i%4):
            b = b.rotate()
        isomorphic_pattern.append(np.array(b.tile)[pattern])
        
    return isomorphic_pattern

class TuplesNet():
    def __init__(self, pattern, maxnum):
        self.V = np.zeros(([maxnum]*len(pattern)))
        self.pattern = pattern
        self.isomorphic_pattern = find_isomorphic_pattern(self.pattern)
        
    def getState(self, tile):
        return [tuple(np.array(tile)[p]) for p in self.isomorphic_pattern]
    
    def getValue(self, tile):
        S = self.getState(tile)
        
        V = [self.V[s] for s in S]
        
        # sum all value from isomorphic pattern
        V = sum(V)

        return V
    
    def setValue(self, tile, v, reset=False):
        S = self.getState(tile)

        v /= len(self.isomorphic_pattern)
        V = 0.0
        for s in S:
            if not reset:
                # update value to isomorphic pattern
                self.V[s] += v
            else:
                # reset value to isonorphic pattern
                self.V[s] =  v
                
            V += self.V[s]
        return V

class Agent():
    def __init__(self, patterns, maxnum):
        self.Tuples = []
        for p in patterns:
            self.Tuples.append(TuplesNet(p, maxnum))
        self.metrics = []
        # if True, use after-state. Otherwise use before-state
        self.after = True
        
    def getValue(self, tile):
        return sum([t.getValue(tile) for t in self.Tuples])
    
    def setValue(self, tile, v, reset=False):
        v /= len(self.Tuples)
        V = 0.0
        for t in self.Tuples:
            V += t.setValue(tile, v, reset)
        return V
    
    # get all s' and reward in next_games
    def evaluate(self, next_games):
        # TD(0)-after
        if self.after:    
            #  r + V(s')
            return [ng[1] + self.getValue(ng[0].tile) for ng in next_games]
        # TD(0)-before
        else:
            # r + \sum P(s''|s')V(s'')
            rs = []
            for ng in next_games:
                all_v = [ self.getValue(nng.tile) for nng in ng[0].allpopup() ]
                if len(all_v) == 0:
                    v = 0
                else:
                    v = sum(all_v) / len(all_v)
                rs.append(ng[1] + v)
            return rs
    
    def learn(self, records, lr):
        # learn from end to begin
        # records = [end .... begin]
        # (s, a, r, s', s'')
        
        # TD(0)-after
        if self.after:
            exact = 0.0
            for s, a, r, s_, s__ in records: 
                # V(s') = V(s') + \alpha ( r_next + V(s'_next) - V(s') )
                error = exact - self.getValue(s_)
                exact = r + self.setValue(s_, lr*error)
        # TD(0)-before
        else:
            #exact = self.getValue(records[0][4])
            exact = 0.0
            for s, a, r, s_, s__ in records:
                # V(s) = V(s) + \alpha (r + V(s'') - V(s))
                error = r + exact - self.getValue(s)
                exact = r + self.setValue(s, lr*error)
            
    def showStattistic(self, epoch, unit, show=True):
        metrics = np.array(self.metrics[epoch-unit:epoch])
        
        # get average score
        score_mean = np.mean(metrics[:, 0])
        # get max score
        score_max = np.max(metrics[:, 0])
        
        if show:
            print('{:<8d}mean = {:<8.0f} max = {:<8.0f}'.format(epoch, score_mean, score_max))
        
        if (metrics.shape[1] < 3):
            return score_mean, score_max
        
        # all end game board
        end_games = metrics[:, 2]
        
        reach_nums = np.array([1<<max(end) & -2 for end in end_games])
                  
        if show:
            print('\n')
        
        score_stat = []
        
        for num in np.sort(np.unique(reach_nums)):
            # count how many game over this num
            reachs = np.count_nonzero(reach_nums >= num)
            reachs = (reachs*100)/len(metrics)
            
            # count how many game end at this num
            ends = np.count_nonzero(reach_nums == num)
            ends = (ends*100)/len(metrics)
            
            if show:
                print('{:<5d}  {:3.1f} % ({:3.1f} %)'.format(num, reachs, ends) )
            
            score_stat.append( (num, reachs, ends) )
        
        score_stat = np.array(score_stat)
        
        return score_mean, score_max, score_stat
    
    def train(self, epoch_size, lr=0.1, showsize=1000):
        start_epoch = len(self.metrics)
        for epoch in range(start_epoch, epoch_size):
            # init score and env (2048)
            score = 0.0
            game = board().popup().popup()
            records = []
            while True:
                # choose action
                next_games = [game.up(), game.down(), game.left(), game.right()]
                action = np.argmax(self.evaluate(next_games))
                
                # do action
                # s'
                next_game, reward = next_games[action]
                
                # save record (s, a, r, s')
                # records.insert(0, (game.tile, action, reward, next_game.tile) )
                
                # if game is same as before, end game
                if game.end():
                    break
                
                # s''
                next_game_after = next_game.popup()
                
                score += reward
                
                # save record (s, a, r, s', s'')
                records.insert(0, (game.tile, action, reward, next_game.tile, next_game_after.tile) )
                
                # s = s'' update state
                game = next_game_after
                
            #self.learn(records, lr / len(self.Tuples))
            self.learn(records, lr)
            
            # store score, game len, end game board
            self.metrics.append( (score, len(records), game.tile) )
            if (epoch+1) % showsize == 0:
                clear_output(wait=True)
                self.showStattistic(epoch+1, showsize)
    
    # use current state of game, return next game and action
    def play(self, game):
        next_games = [game.up(), game.down(), game.left(), game.right()]
        action = np.argmax(self.evaluate(next_games))
                
        next_game, reward = next_games[action]
        return next_game, reward, ['up', 'down', 'left', 'right'][action]

def saveAgent(agent, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(agent, f)
    return fileName
    
def loadAgent(fileName):
    with open(fileName, 'rb') as f:
        agent = pickle.load(f)
    return agent
    
MAX_NUM = 15 # 1<<15 == 32768
TUPLE_NUM = 6 # 6-tuples
PATTERN_NUM = 4
ACTION_NUM = 4 # up, down, left, right

PATTERNS = [
    [0,1,2,3,4,5],
    [4,5,6,7,8,9],
    [0,1,2,4,5,6],
    [4,5,6,8,9,10]
]

random.seed(756110)
agent = Agent(PATTERNS, MAX_NUM)

agent.train(100000)

def showCurve(metrics, size=1000):
    l = (len(metrics) // size) * size
    metrics = np.array(metrics[:l]).reshape(l, -1)
    
    size = int(size)
    
    scores = np.vsplit(metrics[:, 0:1], metrics.shape[0] / size)
    
    scores_mean = [np.mean(s) for s in scores]
    scores_max = [np.max(s) for s in scores]
    scores_min = [np.min(s) for s in scores]
    
    plt.figure(figsize=(12,4))
    #plt.plot(metrics[:,0], '.', label='score')
    plt.plot(scores_mean, label='mean of {:d} epochs'.format(size))
    plt.plot(scores_max, label='max of {:d} epochs'.format(size))
    plt.plot(scores_min, label='min of {:d} epochs'.format(size))
    plt.legend()
    plt.ylabel('2048 score')
    plt.xlabel('every {:d} epochs'.format(size))
    
def showWinRate(metrics, nums, size=1000):
    l = (len(metrics) // size) * size
    metrics = np.array(metrics[:l]).reshape(l, -1)
    
    size = int(size)
    
    reach_nums = np.array([1<<max(end) & -2 for end in metrics[:,2]]).reshape(-1, 1)
    reach_nums = np.vsplit(reach_nums, reach_nums.shape[0] / size)
    
    # (index, nums)
    win_rates = np.array([[np.count_nonzero(rn >= num) / rn.shape[0] for num in nums] for rn in reach_nums])
    
    plt.figure(figsize=(12,4))
    for i, num in enumerate(nums):
        plt.plot(win_rates[:, i], label='win rate of {:d}'.format(int(num)))
    plt.legend()
    plt.xlabel('every {:d} epochs'.format(size))
    plt.ylabel('win rate')
    
showCurve(agent.metrics, size=1000)
showWinRate(agent.metrics, [1024, 2048, 4096])

def playWithAgent(agent, step_per_seconds=0.5, show=True):
    game = board().popup().popup()
    score = 0.0
    step = 0.0
    while not game.end():
        if show:
            clear_output(wait=True)
            print('Score : {:10.0f} Step : {:10.0f}'.format(score, step))
            print(game)
        
        start = time.time()
        next_game, reward, action = agent.play(game)
        while time.time() - start < step_per_seconds:
            pass
        game = next_game.popup()
        if reward < 0.0:
            reward = 0.0
        score += reward
        step += 1.0
    
    return score, step, game.tile

def testAgent(agent, times):
    metrics = []
    for i in range(times):
        metrics.append( playWithAgent(agent, step_per_seconds=0.0, show=False) )
        clear_output(wait=True)
        print('prograss : {:^8d}/{:^8d}'.format(i+1, times))
    
    clear_output(wait=True)
    metrics = np.array(metrics)
    # get average score
    score_mean = np.mean(metrics[:, 0])
    # get max score
    score_max = np.max(metrics[:, 0])
    # all end game board
    end_games = metrics[:, 2]
    reach_nums = np.array([1<<max(end) & -2 for end in end_games])
    score_stat = []

    print('times of play = {:<8d} mean = {:<8.0f} max = {:<8.0f}\n'.format(len(metrics), score_mean, score_max))

    for num in np.sort(np.unique(reach_nums)):
        # count how many game over this num
        reachs = np.count_nonzero(reach_nums >= num)
        reachs = (reachs*100)/len(metrics)
            
        # count how many game end at this num
        ends = np.count_nonzero(reach_nums == num)
        ends = (ends*100)/len(metrics)
            
        print('{:<5d}  {:3.1f} % ({:3.1f} %)'.format(num, reachs, ends) )
            
        score_stat.append( (num, reachs, ends) )
    
    print('\n')
    
    score_stat = np.array(score_stat)
    
    return score_mean, score_max, score_stat

testAgent(agent, 1000)