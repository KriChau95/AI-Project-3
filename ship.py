# ship.py represents all the information to simulate the space vessel and contains bot functions for each bot

# Importing libraries for randomness, data structures, and data visualization
import random
import heapq
from collections import deque, defaultdict
import copy
import math
from visualize import *
from tqdm import tqdm

# setting up global variables that are used for adjacency in searches
global directions
directions = [(0,1), (0,-1), (1,0), (-1,0)] # array to store adjacent directions needed during various traversal
global diagonal_directions
diagonal_directions = [(0,1), (0,-1), (1,0), (-1,0), (-1,1), (-1,-1), (1,1), (1,-1)]

# function to initialize ship by creating maze structure and randomly placing bot and rat
def init_ship(dimension):
    info = dict()

    # 0 = open cell
    # 1 = closed cell
    # 2 = bot

    d = dimension - 2

    # initialize ship to size dimension
    ship = [[1] * d for _ in range(d)] 

    # open up a random cell on the interior
    to_open = random.sample(range(1,d-1), 2)
    row, col = to_open
    ship[row][col] = 0

    single_neighbor = set() # stores all cells' blocked coordinates that have exactly 1 open neighbor
    closed = set() # stores cells that have no chance for being blocked coordinates with exactly 1 open neighbor

    # initialize single_neighbor set based on first open cell
    for dr, dc in directions:
        r = row + dr
        c = col + dc
        if 0 <= r < d and 0 <= c < d:
            single_neighbor.add((r,c))

    # Iteratively opening up cells to create maze structure
    while single_neighbor:

        chosen_coordinate = random.choice(list(single_neighbor)) # choose cell randomly
        single_neighbor.remove(chosen_coordinate) # once cell is open, it can no longer be a blocked cell
        
        row, col = chosen_coordinate 
        ship[row][col] = 0 # open it up
        
        # determine which cells are new candidates to be single neighbors and add cells that have already been dealt with to a closed set
        for dr,dc in directions:
            r = row + dr
            c = col + dc
            if 0 <= r < d and 0 <= c < d and ship[r][c] == 1 and (r,c) not in closed:
                if (r,c) in single_neighbor:
                    single_neighbor.remove((r,c))
                    closed.add((r,c))
                else:
                    single_neighbor.add((r,c))
    
    # Identifying and handling deadend cells
    
    deadends = dict()

    # deadends = open cells with exactly 1 open neighbor
    # deadends dictionary:
    # key: (r,c) s.t. (r,c) is an open cell with exactly 1 open neighbor
    # value: list of (r,c) tuples that represent key's closed neighbors
    
    for r in range(d):
        for c in range(d):
            if ship[r][c] == 0: # open cell
                open_n_count = 0
                closed_neighbors = []
                for dr,dc in directions:
                    nr,nc = r + dr, c + dc
                    if 0 <= nr < d and 0 <= nc < d:
                        if ship[nr][nc] == 0: # open neighbor
                            open_n_count += 1
                        elif ship[nr][nc] == 1:
                            closed_neighbors.append((nr,nc))
                if open_n_count == 1:
                    deadends[(r,c)] = closed_neighbors

    # for ~ 1/2 of deadend cells, pick 1 of their closed neighbors at random and open it
    for i in range(len(deadends)//2):
        list_closed_neighbors = deadends.pop(random.choice(list(deadends.keys()))) # retrieve a random deadend cell's list of closed neighbors
        r,c = random.choice(list_closed_neighbors) # choose a random closed neighbor
        ship[r][c] = 0 # open it

    # ensure border is closed
    ship.insert(0,[1] * dimension)
    ship.append([1] * dimension)
    for i in range(1,dimension-1):
        row = ship[i]
        new_row = [1] + row + [1]
        ship[i] = new_row
    
    # determine remaining open cells
    open_cells = set()
    for r in range(dimension):
        for c in range(dimension):
            if ship[r][c] == 0:
                open_cells.add((r,c))

    # Condense all the information created within this function into a hashmap and return the hashmap

    empty_ship = copy.deepcopy(ship)

    info['open_cell'] = open_cells
    bot_r,bot_c = random.choice(list(open_cells))
    open_cells.remove((bot_r,bot_c))

    ship[bot_r][bot_c] = 2

    info['ship'] = ship
    info['empty_ship'] = empty_ship
    info['bot'] = (bot_r, bot_c)
    info['open_cell'] = open_cells
    return info

def bot(info, visualize):

    empty_ship = info['empty_ship']
    ship = info['ship']
    d = len(ship)
    possible_ship = copy.deepcopy(empty_ship)
    set_possible_cells = set()
    open_cells = info['open_cell']

    if visualize:
        visualize_ship(ship, None, title = "Initial Ship")

    # initialize set of possible cells - all open cells

    for i in range(d):
        for j in range(d):
            if empty_ship[i][j] == 0:
                possible_ship[i][j] = 2
                set_possible_cells.add((i,j))
    
    if visualize:
        visualize_possible_cells(empty_ship, set_possible_cells, title = "Visualize Possible Cells")
        visualize_ship(possible_ship, None, title = "Possible Ship")

    
    # dead end: three blocked neighbors
    # corner: two adjacent directions are blocked (left/down, right/up, right/down, left/up)

    set_targets = set()

    for cell in set_possible_cells:
        blocked = []
        for dr, dc in directions:
            nr, nc = cell[0] + dr, cell[1] + dc
            if empty_ship[nr][nc] == 1:
                blocked.append((dr, dc))
        if len(blocked) == 3:
            set_targets.add(cell)
        elif len(blocked) == 2:
            if blocked[0][0] + blocked[1][0] != 0:
                set_targets.add(cell)

    num_samples = 100
    histogram = dict()
    for L_size in tqdm(range(len(set_possible_cells), 1, -1)):
        avg = 0
        for i in range(num_samples):
            sample = random.sample(list(set_possible_cells), L_size)
            avg += run(set_targets, empty_ship, set(sample), False)
        avg/=num_samples
        histogram[L_size] = avg

    with open("localization_results.txt", "w") as f:
        for L_size in sorted(histogram.keys(), reverse=True):
            avg_moves = histogram[L_size]
            f.write(f"L_size = {L_size}, Average Moves = {avg_moves:.2f}\n")


    # L_size = len(set_possible_cells) - 20
    # sample = random.sample(list(set_possible_cells), L_size)

    # if visualize:
    #     visualize_possible_cells(ship, sample, title = "sample of L - 20")
                           
            
    # if visualize:
    #     visualize_possible_cells(empty_ship, set_targets, title = "Set Targets")
    
    print(histogram)


def run(set_targets, empty_ship, set_possible_cells, visualize):
    iterations = 0
    while len(set_possible_cells) > 1:
        target_cell = random.choice(list(set_targets))
        start_cell = random.choice(list(set_possible_cells))
        path = astar(start_cell,empty_ship,target_cell)
        prev_cell = start_cell
        for i in range(len(path)):
            curr_cell = path[i]
            move = (curr_cell[0]-prev_cell[0], curr_cell[1] - prev_cell[1])
            new_set_possible_cells = update_location_set(set_possible_cells, empty_ship, move)
            if visualize:
                visualize_possible_cells(empty_ship, new_set_possible_cells, title=f"set_possible_cells is size {len(set_possible_cells)} at iteration {iterations} after moving {move}")
            set_possible_cells = new_set_possible_cells
            if len(set_possible_cells) == 1:
                break
            prev_cell = curr_cell
            iterations +=1
    return iterations

def update_location_set(L, ship, move):

    new_L = set()

    for possible_cell in L:
        
        new_r = possible_cell[0] + move[0]
        new_c = possible_cell[1] + move[1]
        
        if ship[new_r][new_c] == 1: # next position if moved, but ran into wall
            new_L.add(possible_cell)
        else:
            new_L.add((new_r, new_c)) # next position if moved, successfully

    return new_L


# A* algorithm to find shortest path from any start position to any end position 
def astar(start, map, end):
    def heuristic(cell1):
        return abs(cell1[0] - end[0]) + abs(cell1[1]-end[1]) 
    
    d = len(map)
    fringe = []
    heapq.heappush(fringe, (heuristic(start),start))
    total_costs = dict()
    total_costs[start] = 0
    prev = dict()
    prev[start] = None

    while fringe:
        curr = heapq.heappop(fringe)
        if curr[1] == end:
            curr_p = curr[1]
            path = deque()
            while curr_p != None:
                path.appendleft(curr_p)
                curr_p = prev[curr_p]
            return list(path)
        
        r,c = curr[1]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            child = (nr,nc)
            if 0 <= nr < d and 0 <= nc < d and (map[nr][nc] != 1 and map[nr][nc] != -1):
                cost = total_costs[curr[1]] + 1
                est_total_cost = cost + heuristic(child)
                if child not in total_costs:
                    prev[child] = curr[1]
                    total_costs[child] = cost
                    heapq.heappush(fringe, (est_total_cost, child))
    return []        


def main():
    random.seed(21)

    og_info = init_ship(30)
    # visualize_ship(og_info['ship'], None)

    random.seed()

    iterations = bot(og_info, visualize = False)
    print(iterations)

if __name__ == "__main__":
    main()