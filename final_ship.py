# ship.py represents all the information to simulate the space vessel and contains bot functions for each bot

# Importing libraries for randomness, data structures, and data visualization
import random
import heapq
from collections import deque, defaultdict
import copy
import math
from visualize import *
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    import pickle

    empty_ship = info['empty_ship']
    ship = info['ship']
    d = len(ship)
    possible_ship = copy.deepcopy(empty_ship)
    set_possible_cells = set()
    open_cells = info['open_cell']

    if visualize:
        visualize_ship(ship, None, title="Initial Ship")

    # Initialize set of possible cells
    for i in range(d):
        for j in range(d):
            if empty_ship[i][j] == 0:
                possible_ship[i][j] = 2
                set_possible_cells.add((i, j))

    if visualize:
        visualize_possible_cells(empty_ship, set_possible_cells, title="Visualize Possible Cells")
        visualize_ship(possible_ship, None, title="Possible Ship")

    # Identify target locations (deadends + corners)
    set_targets = set()
    for cell in set_possible_cells:
        blocked = []
        for dr, dc in directions:
            nr, nc = cell[0] + dr, cell[1] + dc
            if empty_ship[nr][nc] == 1:
                blocked.append((dr, dc))
        if len(blocked) == 3:
            set_targets.add(cell)

    rows, cols = len(empty_ship), len(empty_ship[0])
    corner_coords = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]

    def find_nearest_open(r0, c0):
        visited = set()
        q = deque([(r0, c0)])
        while q:
            next_layer = deque()
            result = []
            for _ in range(len(q)):
                r, c = q.popleft()
                if 0 <= r < rows and 0 <= c < cols and empty_ship[r][c] == 0:
                    result.append((r, c))
                else:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            next_layer.append((nr, nc))
            if result:
                return result
            q = next_layer
        return []

    for r0, c0 in corner_coords:
        for oc in find_nearest_open(r0, c0):
            set_targets.add(oc)

    if visualize:
        visualize_possible_cells(ship, set_targets, "Deadends + Corners")

    # === Initialization ===
    avg_remaining_moves_map = defaultdict(lambda: [0, 0])
    data = []
    labels = []

    # === Phase 1: Natural random targets ===
    # print("Phase 1: Random target selection")
    num_runs = 100
    # for _ in tqdm(range(num_runs)):
    #     for _ in range(3):
    #         target_cell = random.choice(list(set_targets))
    #         map_snaps, L_list, iterations = run(target_cell, empty_ship, set_possible_cells.copy(), visualize)
    #         if iterations >= 1000:
    #             continue
    #         remaining_moves = list(range(len(L_list) - 1, -1, -1))
    #         data.extend(map_snaps)
    #         labels.extend(remaining_moves)

    #         for item in set(L_list):
    #             first_occ = L_list.index(item)
    #             last_occ = len(L_list) - L_list[::-1].index(item)
    #             midpoint = first_occ + (last_occ - first_occ) // 2
    #             remaining = len(L_list) - 1 - midpoint
    #             avg_remaining_moves_map[item][0] += remaining
    #             avg_remaining_moves_map[item][1] += 1

    # === Phase 2: Fixed |L| sampling ===
    print("Phase 2: Fixed L size")
    target_L_sizes = range(300, 546, 3)
    samples_per_L = 10
    histogram = dict()

    for L_size in tqdm(target_L_sizes):
        if L_size > len(open_cells):
            continue

        total_moves = 0
        for _ in range(samples_per_L):
            target_cell = random.choice(list(set_targets))
            possible_sample = set(random.sample(list(open_cells), L_size))
            map_snaps, L_list, iterations = run(target_cell, empty_ship, possible_sample, visualize=False)


            if iterations >= 1000:
                continue
            total_moves += iterations
            remaining_moves = list(range(len(L_list) - 1, -1, -1))
            data.extend(map_snaps)
            labels.extend(remaining_moves)

            for item in set(L_list):
                first_occ = L_list.index(item)
                last_occ = len(L_list) - L_list[::-1].index(item)
                midpoint = first_occ + (last_occ - first_occ) // 2
                remaining = len(L_list) - 1 - midpoint
                avg_remaining_moves_map[item][0] += remaining
                avg_remaining_moves_map[item][1] += 1

        histogram[L_size] = total_moves / samples_per_L

    # === Finalize average map ===
    for L_size in avg_remaining_moves_map:
        total, count = avg_remaining_moves_map[L_size]
        avg_remaining_moves_map[L_size] = total / count

    # === Save outputs ===
    with open("avg_remaining_moves.txt", "w") as f:
        for L_size in sorted(avg_remaining_moves_map):
            f.write(f"{L_size}: {avg_remaining_moves_map[L_size]:.4f}\n")

    with open("localization_results.txt", "w") as f:
        for L_size in sorted(histogram.keys()):
            f.write(f"{L_size}: {histogram[L_size]:.2f}\n")

    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)

    with open("labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    print(f"Saved {len(data)} data samples and labels.")

def bot_2(info, visualize, model):
    import pickle

    empty_ship = info['empty_ship']
    ship = info['ship']
    d = len(ship)
    possible_ship = copy.deepcopy(empty_ship)
    set_possible_cells = set()
    open_cells = info['open_cell']

    if visualize:
        visualize_ship(ship, None, title="Initial Ship")

    # Initialize set of possible cells
    for i in range(d):
        for j in range(d):
            if empty_ship[i][j] == 0:
                possible_ship[i][j] = 2
                set_possible_cells.add((i, j))

    if visualize:
        visualize_possible_cells(empty_ship, set_possible_cells, title="Visualize Possible Cells")
        visualize_ship(possible_ship, None, title="Possible Ship")

    # Identify target locations (deadends + corners)
    set_targets = set()
    for cell in set_possible_cells:
        blocked = []
        for dr, dc in directions:
            nr, nc = cell[0] + dr, cell[1] + dc
            if empty_ship[nr][nc] == 1:
                blocked.append((dr, dc))
        if len(blocked) == 3:
            set_targets.add(cell)

    rows, cols = len(empty_ship), len(empty_ship[0])
    corner_coords = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]

    def find_nearest_open(r0, c0):
        visited = set()
        q = deque([(r0, c0)])
        while q:
            next_layer = deque()
            result = []
            for _ in range(len(q)):
                r, c = q.popleft()
                if 0 <= r < rows and 0 <= c < cols and empty_ship[r][c] == 0:
                    result.append((r, c))
                else:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            next_layer.append((nr, nc))
            if result:
                return result
            q = next_layer
        return []

    for r0, c0 in corner_coords:
        for oc in find_nearest_open(r0, c0):
            set_targets.add(oc)

    if visualize:
        visualize_possible_cells(ship, set_targets, "Deadends + Corners")

    # === Initialization ===
    avg_remaining_moves_map = defaultdict(lambda: [0, 0])
    data = []
    labels = []

    avg_remaining_moves_map_2 = defaultdict(lambda: [0, 0])
    data_2 = []
    labels_2 = []

    # === Phase 1: Natural random targets ===
    # print("Phase 1: Random target selection")
    num_runs = 100
   
    print("Phase 2: Fixed L size")
    target_L_sizes = range(2, 500, 5)
    samples_per_L = 10
    histogram = dict()

    for L_size in tqdm(target_L_sizes):

        if L_size > len(open_cells):
            continue

        total_moves = 0
        for _ in range(samples_per_L):

            # print("L_size:", L_size, "sample:", _)

            target_cell = random.choice(list(set_targets))
            possible_sample = set(random.sample(list(open_cells), L_size))

            # print("possible_sample:", possible_sample)

            best_dir = (0, 0)

            lowest_moves = float('inf')
            metadata = torch.load("model_metadata.pth")
            y_min, y_max = metadata["y_min"], metadata["y_max"]
            

            for move in directions:

                moved_sample = update_location_set(possible_sample, empty_ship, move)
                # visualize_possible_cells(empty_ship, moved_sample, title = f"Move {move}")
                
                map_copy = copy.deepcopy(empty_ship)
                for r, c in moved_sample:
                    map_copy[r][c] = -1

                input_tensor = torch.tensor(np.array(map_copy), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    normalized_pred = model(input_tensor).item()
                    prediction = normalized_pred * (y_max - y_min) + y_min
                    # print(prediction, move)
                    if prediction < lowest_moves:
                        lowest_moves = prediction
                        best_sample = moved_sample



            # print(L_size)
            # visualize_possible_cells(empty_ship, possible_sample, title = f"Actual Initial State")
            # visualize_possible_cells(empty_ship, best_sample, title = f"After Moving State")

            map_snaps, L_list, iterations = run(target_cell, empty_ship, possible_sample, visualize=False,bot="bot1")
            map_snaps_2, L_list_2, iterations_2 = run(target_cell, empty_ship, best_sample, visualize=False,bot="bot2")

            start_map = [map_snaps[0]] 
            start_num_open_cells = L_list[0] 
            final_iterations = 1 + iterations_2

            # print(f"bot1: {iterations}, bot2: {iterations_2+1}")


            
            if iterations >= 1000 or iterations_2 >= 1000:
                continue

            total_moves += iterations
            remaining_moves = list(range(len(L_list) - 1, -1, -1))
            data.extend(map_snaps)
            labels.extend(remaining_moves)

            for item in set(L_list):

                first_occ = L_list.index(item)
                last_occ = len(L_list) - L_list[::-1].index(item)
                midpoint = first_occ + (last_occ - first_occ) // 2
                remaining = len(L_list) - 1 - midpoint
                avg_remaining_moves_map[item][0] += remaining
                avg_remaining_moves_map[item][1] += 1

            
            avg_remaining_moves_map_2[start_num_open_cells][0] += final_iterations
            avg_remaining_moves_map_2[start_num_open_cells][1] += 1


        histogram[L_size] = total_moves / samples_per_L

    # === Finalize average map ===
    for L_size in avg_remaining_moves_map:
        total, count = avg_remaining_moves_map[L_size]
        avg_remaining_moves_map[L_size] = total / count

    for L_size in avg_remaining_moves_map_2:
        total, count = avg_remaining_moves_map_2[L_size]
        avg_remaining_moves_map_2[L_size] = total / count

    # === Save outputs ===
    with open("avg_remaining_moves.txt", "w") as f:
        for L_size in sorted(avg_remaining_moves_map):
            f.write(f"{L_size}: {avg_remaining_moves_map[L_size]:.4f}\n")
    
    with open("avg_remaining_moves_2.txt", "w") as f:
        for L_size in sorted(avg_remaining_moves_map_2):
            f.write(f"{L_size}: {avg_remaining_moves_map_2[L_size]:.4f}\n")

    with open("localization_results.txt", "w") as f:
        for L_size in sorted(histogram.keys()):
            f.write(f"{L_size}: {histogram[L_size]:.2f}\n")

    # with open("data.pkl", "wb") as f:
    #     pickle.dump(data, f)

    # with open("labels.pkl", "wb") as f:
        # pickle.dump(labels, f)

    print(f"Saved {len(data)} data samples and labels.")


def run(target_cell, empty_ship, set_possible_cells, visualize, bot):
    
    map_snaps = [] # snapshot of map at current time state
    L_list = [] # labels
    iterations = 0
    

    while len(set_possible_cells) > 1 and iterations < 1000:    
        # print(len(set_possible_cells))
        start_cell = random.choice(list(set_possible_cells))
        path = astar(start_cell,empty_ship,target_cell)
        prev_cell = start_cell
        for i in range(len(path)):
            L_list.append((len(set_possible_cells)))
            
            # store input for test data
            curr_map = copy.deepcopy(empty_ship)
            for r in range(len(curr_map)):
                for c in range(len(curr_map)):
                    if (r,c) in set_possible_cells:
                        curr_map[r][c] = -1
            
            map_snaps.append(curr_map)

            # print("Here:", iterations, L_list)
            curr_cell = path[i]
            move = (curr_cell[0]-prev_cell[0], curr_cell[1] - prev_cell[1])
            new_set_possible_cells = update_location_set(set_possible_cells, empty_ship, move)
            if visualize and len(set_possible_cells) == 2:
                visualize_possible_cells(empty_ship, new_set_possible_cells, title=f"set_possible_cells is size {len(new_set_possible_cells)} at iteration {iterations} after moving {move}")
            set_possible_cells = new_set_possible_cells
            prev_cell = curr_cell
            iterations +=1
            if len(set_possible_cells) == 1:
                break
            if iterations == 1000:
                break
    L_list.pop(0)
    map_snaps.pop(0)
    L_list.append(1)
    last_map = copy.deepcopy(empty_ship)
    fr, fc = next(iter(set_possible_cells))
    last_map[fr][fc] = -1
    map_snaps.append(last_map)

    # print(len(L_list))
    # print(len(map_snaps))

    return map_snaps, L_list, iterations


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

    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()

            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride = 1, padding=1)  
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding=1)  
            self.pool = nn.MaxPool2d(2, 2) 
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride = 1, padding=1) 
            self.fc1 = nn.Linear(128 * 15 * 15, 256)  
            self.fc2 = nn.Linear(256, 1)  

        def forward(self, x):
            x = F.relu(self.conv1(x)) 
            x = self.pool(F.relu(self.conv2(x)))  
            x = F.relu(self.conv3(x))  
            x = x.view(x.size(0), -1)  
            x = F.relu(self.fc1(x))  
            x = self.fc2(x) 
            return x
    
    model = CNNModel()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()



    # visualize_ship(og_info['ship'], None)

    # bot(og_info, visualize = False)
    bot_2(og_info, visualize = False, model = model)

if __name__ == "__main__":
    main()