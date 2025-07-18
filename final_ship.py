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

# array to store adjacent directions needed during various traversal
global directions
directions = [(0,1), (0,-1), (1,0), (-1,0)]
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

    # Initialize variables and extract relevant information from info
    empty_ship = info['empty_ship']
    ship = info['ship']
    d = len(ship)
    possible_ship = copy.deepcopy(empty_ship)
    set_possible_cells = set()
    open_cells = info['open_cell']

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

    # Find dead ends
    set_targets = set()
    for cell in set_possible_cells:
        blocked = []
        for dr, dc in directions:
            nr, nc = cell[0] + dr, cell[1] + dc
            if empty_ship[nr][nc] == 1:
                blocked.append((dr, dc))
        if len(blocked) == 3:
            set_targets.add(cell)


    # Finding corners by choosing closest open cells to the corner of the ship
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

    
    # Main loop: initialize data and label arrays, and start generating samples
    avg_remaining_moves_map = defaultdict(int)
    data = []
    labels = []

    # Generating 150 samples for each L size
    target_L_sizes = list(range(2,600,1))
    samples_per_L = 150

    for L_size in tqdm(target_L_sizes):
        
        num_samples = 0

        if L_size > len(open_cells):
            break

        while num_samples < samples_per_L:

            target = random.choice(list(set_targets))
            
            L_initial = set(random.sample(list(open_cells), L_size))

            map_snaps, L_list, iterations = run(target, empty_ship, L_initial, visualize = False)

            start_map = map_snaps[0]
            
            if iterations >= 1000:
                continue
                
            num_samples += 1

            data.append(start_map)
            labels.append(iterations)

            avg_remaining_moves_map[L_size] += iterations

    # Averaging generated data 
    for L_size in avg_remaining_moves_map:
        total = avg_remaining_moves_map[L_size]
        avg_remaining_moves_map[L_size] = total / samples_per_L

    # Save data
    with open("avg_remaining_moves.txt", "w") as f:
        for L_size in sorted(avg_remaining_moves_map):
            f.write(f"{L_size}: {avg_remaining_moves_map[L_size]:.4f}\n")

    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)

    with open("labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    print(f"Saved {len(data)} data samples and labels.")


# Generate Policy 1 data. Takes in model trained on Policy 0 (base bot)
def bot_2(info, visualize, model):

    # Initialize variables and extract relevant information from info
    empty_ship = info['empty_ship']
    ship = info['ship']
    d = len(ship)
    possible_ship = copy.deepcopy(empty_ship)
    set_possible_cells = set()
    open_cells = info['open_cell']

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

    # Find dead ends
    set_targets = set()
    for cell in set_possible_cells:
        blocked = []
        for dr, dc in directions:
            nr, nc = cell[0] + dr, cell[1] + dc
            if empty_ship[nr][nc] == 1:
                blocked.append((dr, dc))
        if len(blocked) == 3:
            set_targets.add(cell)


    # Finding corners by choosing closest open cells to the corner of the ship
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

    # Initialize avg remaining move map to compare Policy 1 to Policy 0
    avg_remaining_moves_map = defaultdict(int)
    avg_remaining_moves_map_2 = defaultdict(int)


    # Main loop: initialize data and label arrays, and start generating samples
    data = []
    labels = []

    data_2 = []
    labels_2 = []

    target_L_sizes = list(range(2,600,1))

    # Generating 100 samples for each L size
    samples_per_L = 100

    for L_size in tqdm(target_L_sizes):

        if L_size > len(open_cells):
            break

        num_samples = 0

        while num_samples < samples_per_L:

            target = random.choice(list(set_targets))
            
            L_initial = set(random.sample(list(open_cells), L_size))
            
            best_dir = (0, 0)

            lowest_moves = float('inf')

            for move in directions:

                moved_sample = update_location_set(L_initial, empty_ship, move)
                # visualize_possible_cells(empty_ship, moved_sample, title = f"Move {move}")
                
                map_copy = copy.deepcopy(empty_ship)
                for r, c in moved_sample:
                    map_copy[r][c] = -1

                input_tensor = torch.tensor(np.array(map_copy), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = model(input_tensor).item() 
                    # print(prediction, move)
                    if prediction < lowest_moves:
                        lowest_moves = prediction
                        best_sample = moved_sample

            map_snaps, L_list, iterations = run(target, empty_ship, L_initial, visualize = False)
            map_snaps_2, L_list_2, iterations_2 = run(target, empty_ship, best_sample, visualize=False)
            iterations_2+=1

            start_map = map_snaps[0]
            
            if iterations >= 1000 or iterations_2>=1000:
                continue
                
            num_samples += 1

            data.append(start_map)
            labels.append(iterations)

            data_2.append(start_map)
            labels_2.append(iterations_2)

            avg_remaining_moves_map[L_size] += iterations
            avg_remaining_moves_map_2[L_size] += iterations_2

    # Averaging generated data 
    for L_size in avg_remaining_moves_map:
        total = avg_remaining_moves_map[L_size]
        avg_remaining_moves_map[L_size] = total / samples_per_L

    for L_size in avg_remaining_moves_map_2:
        total = avg_remaining_moves_map_2[L_size]
        avg_remaining_moves_map_2[L_size] = total / samples_per_L

    # Save data
    with open("avg_remaining_moves.txt", "w") as f:
        for L_size in sorted(avg_remaining_moves_map):
            f.write(f"{L_size}: {avg_remaining_moves_map[L_size]:.4f}\n")

    with open("avg_remaining_moves_2.txt", "w") as f:
        for L_size in sorted(avg_remaining_moves_map_2):
            f.write(f"{L_size}: {avg_remaining_moves_map_2[L_size]:.4f}\n")

    # with open("data.pkl", "wb") as f:
    #     pickle.dump(data, f)

    # with open("labels.pkl", "wb") as f:
    #     pickle.dump(labels, f)

    with open("data_2.pkl", "wb") as f:
        pickle.dump(data_2, f)

    with open("labels_2.pkl", "wb") as f:
        pickle.dump(labels_2, f)

    print(f"Saved {len(data)} data samples and labels.")


# Compare predicted vs actual for our bot
def bot_comp_avg(info, visualize, model):

    empty_ship = info['empty_ship']
    ship = info['ship']
    d = len(ship)
    possible_ship = copy.deepcopy(empty_ship)
    set_possible_cells = set()
    open_cells = info['open_cell']

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

    #Main loop, generate 150 samples per L and compare
    avg_remaining_moves_map = defaultdict(int)

    target_L_sizes = list(range(2,600,1))
    samples_per_L = 150

    for L_size in tqdm(target_L_sizes):

        if L_size > len(open_cells):
            break

        num_samples = 0

        while num_samples < samples_per_L:
            
            L_initial = set(random.sample(list(open_cells), L_size))

            map_copy = copy.deepcopy(empty_ship)
            for r, c in L_initial:
                map_copy[r][c] = -1

            input_tensor = torch.tensor(np.array(map_copy), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
            with torch.no_grad():
                prediction = model(input_tensor).item() 
                
            num_samples += 1

            avg_remaining_moves_map[L_size] += prediction

    # Finalize average map for both
    for L_size in avg_remaining_moves_map:
        total = avg_remaining_moves_map[L_size]
        avg_remaining_moves_map[L_size] = total / samples_per_L

    # Save outputs
    with open("comp_avg_remaining_moves.txt", "w") as f:
        for L_size in sorted(avg_remaining_moves_map):
            f.write(f"{L_size}: {avg_remaining_moves_map[L_size]:.4f}\n")

# Generate Policy 2 data. Takes in model trained on Policy 1 and model trained on Policy 0
def bot_3(info, visualize, model, model_2):

    # Initialize variables and extract relevant information from info
    empty_ship = info['empty_ship']
    ship = info['ship']
    d = len(ship)
    possible_ship = copy.deepcopy(empty_ship)
    set_possible_cells = set()
    open_cells = info['open_cell']


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

    # Find dead ends
    set_targets = set()
    for cell in set_possible_cells:
        blocked = []
        for dr, dc in directions:
            nr, nc = cell[0] + dr, cell[1] + dc
            if empty_ship[nr][nc] == 1:
                blocked.append((dr, dc))
        if len(blocked) == 3:
            set_targets.add(cell)

    # Finding corners by choosing closest open cells to the corner of the ship
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

    # Initialize avg remaining move map to compare Policy 1 to Policy 0
    avg_remaining_moves_map = defaultdict(int)
    avg_remaining_moves_map_2 = defaultdict(int)

    # Main loop: initialize data and label arrays, and start generating samples
    data = []
    labels = []

    data_2 = []
    labels_2 = []

    target_L_sizes = list(range(2,600,2))

    # Generating 100 samples for each L size
    samples_per_L = 100

    for L_size in tqdm(target_L_sizes):

        if L_size > len(open_cells):
            break

        num_samples = 0

        while num_samples < samples_per_L:

            target = random.choice(list(set_targets))
            
            L_initial = set(random.sample(list(open_cells), L_size))

            lowest_moves = float('inf')

            for move in directions:

                moved_sample = update_location_set(L_initial, empty_ship, move)
                # visualize_possible_cells(empty_ship, moved_sample, title = f"Move {move}")
                
                map_copy = copy.deepcopy(empty_ship)
                for r, c in moved_sample:
                    map_copy[r][c] = -1

                input_tensor = torch.tensor(np.array(map_copy), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = model_2(input_tensor).item() 
                    # print(prediction, move)
                    if prediction < lowest_moves:
                        lowest_moves = prediction
                        best_sample = moved_sample

            lowest_moves = float('inf')
            L_initial_2 = best_sample

            for move in directions:

                moved_sample = update_location_set(L_initial_2, empty_ship, move)
                # visualize_possible_cells(empty_ship, moved_sample, title = f"Move {move}")
                
                map_copy = copy.deepcopy(empty_ship)
                for r, c in moved_sample:
                    map_copy[r][c] = -1

                input_tensor = torch.tensor(np.array(map_copy), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = model(input_tensor).item() 
                    # print(prediction, move)
                    if prediction < lowest_moves:
                        lowest_moves = prediction
                        best_sample = moved_sample

            map_snaps, L_list, iterations = run(target, empty_ship, L_initial, visualize = False)
            map_snaps_2, L_list_2, iterations_2 = run(target, empty_ship, best_sample, visualize=False)
            iterations_2+=2

            start_map = map_snaps[0]
            
            if iterations >= 1000 or iterations_2>=1000:
                continue
                
            num_samples += 1

            data.append(start_map)
            labels.append(iterations)

            data_2.append(start_map)
            labels_2.append(iterations_2)

            avg_remaining_moves_map[L_size] += iterations
            avg_remaining_moves_map_2[L_size] += iterations_2

    # Averaging generated data 
    for L_size in avg_remaining_moves_map:
        total = avg_remaining_moves_map[L_size]
        avg_remaining_moves_map[L_size] = total / samples_per_L

    for L_size in avg_remaining_moves_map_2:
        total = avg_remaining_moves_map_2[L_size]
        avg_remaining_moves_map_2[L_size] = total / samples_per_L

    # Save data
    with open("avg_remaining_moves.txt", "w") as f:
        for L_size in sorted(avg_remaining_moves_map):
            f.write(f"{L_size}: {avg_remaining_moves_map[L_size]:.4f}\n")

    with open("avg_remaining_moves_3.txt", "w") as f:
        for L_size in sorted(avg_remaining_moves_map_2):
            f.write(f"{L_size}: {avg_remaining_moves_map_2[L_size]:.4f}\n")

    # with open("data.pkl", "wb") as f:
    #     pickle.dump(data, f)

    # with open("labels.pkl", "wb") as f:
    #     pickle.dump(labels, f)

    # with open("data_2.pkl", "wb") as f:
    #     pickle.dump(data_2, f)

    # with open("labels_2.pkl", "wb") as f:
    #     pickle.dump(labels_2, f)

    with open("data_3.pkl", "wb") as f:
        pickle.dump(data_2, f)

    with open("labels_3.pkl", "wb") as f:
        pickle.dump(labels_2, f)

    print(f"Saved {len(data)} data samples and labels.")


# Generate Policy 3 data. Takes in models trained on Policy 2, Policy 1 and Policy 0
def bot_4(info, visualize, model, model_2, model_3):

    # Initialize variables and extract relevant information from info
    empty_ship = info['empty_ship']
    ship = info['ship']
    d = len(ship)
    possible_ship = copy.deepcopy(empty_ship)
    set_possible_cells = set()
    open_cells = info['open_cell']


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

    # Find dead ends
    set_targets = set()
    for cell in set_possible_cells:
        blocked = []
        for dr, dc in directions:
            nr, nc = cell[0] + dr, cell[1] + dc
            if empty_ship[nr][nc] == 1:
                blocked.append((dr, dc))
        if len(blocked) == 3:
            set_targets.add(cell)

    # Finding corners by choosing closest open cells to the corner of the ship
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

    # Initialize avg remaining move map to compare Policy 1 to Policy 0
    avg_remaining_moves_map = defaultdict(int)
    avg_remaining_moves_map_2 = defaultdict(int)

    # Main loop: initialize data and label arrays, and start generating samples
    data = []
    labels = []

    data_2 = []
    labels_2 = []

    target_L_sizes = list(range(2,600,3))

    # Generating 100 samples for each L size
    samples_per_L = 100

    for L_size in tqdm(target_L_sizes):

        if L_size > len(open_cells):
            break

        num_samples = 0

        while num_samples < samples_per_L:

            target = random.choice(list(set_targets))
            
            L_initial = set(random.sample(list(open_cells), L_size))

            lowest_moves = float('inf')

            for move in directions:

                moved_sample = update_location_set(L_initial, empty_ship, move)
                # visualize_possible_cells(empty_ship, moved_sample, title = f"Move {move}")
                
                map_copy = copy.deepcopy(empty_ship)
                for r, c in moved_sample:
                    map_copy[r][c] = -1

                input_tensor = torch.tensor(np.array(map_copy), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = model_3(input_tensor).item() 
                    # print(prediction, move)
                    if prediction < lowest_moves:
                        lowest_moves = prediction
                        best_sample = moved_sample

            lowest_moves = float('inf')
            L_initial_2 = best_sample

            for move in directions:

                moved_sample = update_location_set(L_initial_2, empty_ship, move)
                # visualize_possible_cells(empty_ship, moved_sample, title = f"Move {move}")
                
                map_copy = copy.deepcopy(empty_ship)
                for r, c in moved_sample:
                    map_copy[r][c] = -1

                input_tensor = torch.tensor(np.array(map_copy), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = model_2(input_tensor).item() 
                    # print(prediction, move)
                    if prediction < lowest_moves:
                        lowest_moves = prediction
                        best_sample = moved_sample

            lowest_moves = float('inf')
            L_initial_3 = best_sample

            for move in directions:

                moved_sample = update_location_set(L_initial_3, empty_ship, move)
                # visualize_possible_cells(empty_ship, moved_sample, title = f"Move {move}")
                
                map_copy = copy.deepcopy(empty_ship)
                for r, c in moved_sample:
                    map_copy[r][c] = -1

                input_tensor = torch.tensor(np.array(map_copy), dtype=torch.float32).unsqueeze(0).unsqueeze(0)  

                with torch.no_grad():
                    prediction = model(input_tensor).item() 
                    # print(prediction, move)
                    if prediction < lowest_moves:
                        lowest_moves = prediction
                        best_sample = moved_sample
                        
            
            map_snaps, L_list, iterations = run(target, empty_ship, L_initial, visualize = False)
            map_snaps_2, L_list_2, iterations_2 = run(target, empty_ship, best_sample, visualize=False)
            iterations_2+=3

            start_map = map_snaps[0]
            
            if iterations >= 1000 or iterations_2>=1000:
                continue
                
            num_samples += 1

            data.append(start_map)
            labels.append(iterations)

            data_2.append(start_map)
            labels_2.append(iterations_2)

            avg_remaining_moves_map[L_size] += iterations
            avg_remaining_moves_map_2[L_size] += iterations_2

    # Averaging generated data 
    for L_size in avg_remaining_moves_map:
        total = avg_remaining_moves_map[L_size]
        avg_remaining_moves_map[L_size] = total / samples_per_L

    for L_size in avg_remaining_moves_map_2:
        total = avg_remaining_moves_map_2[L_size]
        avg_remaining_moves_map_2[L_size] = total / samples_per_L

    # Save data
    with open("avg_remaining_moves.txt", "w") as f:
        for L_size in sorted(avg_remaining_moves_map):
            f.write(f"{L_size}: {avg_remaining_moves_map[L_size]:.4f}\n")

    with open("avg_remaining_moves_4.txt", "w") as f:
        for L_size in sorted(avg_remaining_moves_map_2):
            f.write(f"{L_size}: {avg_remaining_moves_map_2[L_size]:.4f}\n")

    # with open("data.pkl", "wb") as f:
    #     pickle.dump(data, f)

    # with open("labels.pkl", "wb") as f:
    #     pickle.dump(labels, f)

    # with open("data_2.pkl", "wb") as f:
    #     pickle.dump(data_2, f)

    # with open("labels_2.pkl", "wb") as f:
    #     pickle.dump(labels_2, f)

    with open("data_4.pkl", "wb") as f:
        pickle.dump(data_2, f)

    with open("labels_4.pkl", "wb") as f:
        pickle.dump(labels_2, f)

    print(f"Saved {len(data)} data samples and labels.")

# Given a set of possible cells and a target, localizes the robot and returns iterations
def run(target_cell, empty_ship, set_possible_cells, visualize):
    
    # previous approach used map_snaps to track intermediate states
    map_snaps = [] # snapshot of map at current time state
    L_list = [] # labels
    iterations = 0

    curr_map = copy.deepcopy(empty_ship)
    for r in range(len(curr_map)):
        for c in range(len(curr_map)):
            if (r,c) in set_possible_cells:
                curr_map[r][c] = -1
    
    map_snaps.append(curr_map)

    # Choose random start cell, run A* from that cell to target and update map
    # repeat until we are for sure at one specific ell
    while len(set_possible_cells) > 1 and iterations < 1000: # bad targets result in infinite loops resulting in >1000 moves
        
        # choose a random cell to be the start cell to determine a path to target cell
        start_cell = random.choice(list(set_possible_cells))
        path = astar(start_cell,empty_ship,target_cell)
        
        prev_cell = start_cell

        # take the path to the target cell and filter set_possible_cells after each move
        for i in range(len(path)):      

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

    return map_snaps, L_list, iterations


# Updates location set after a move
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

# Model architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 15 * 15, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))      
        x = self.pool(F.relu(self.conv2(x))) 
        x = F.relu(self.conv3(x))       
        x = x.view(x.size(0), -1)       
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    random.seed(21)
    og_info = init_ship(30)
    
    model = CNNModel()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()

    model_2 = CNNModel()
    model_2.load_state_dict(torch.load("model_weights_2.pth"))
    model_2.eval()

    model_3 = CNNModel()
    model_3.load_state_dict(torch.load("model_weights_3.pth"))
    model_3.eval()

    # visualize_ship(og_info['ship'], None)

    # bot(og_info, visualize = True)
    # bot_1(og_info, visualize = True)
    # bot_2(og_info, visualize = False, model = model)
    # bot_3(og_info, visualize = False, model = model, model_2 = model_2)
    bot_4(og_info, visualize = False, model = model, model_2 = model_2, model_3 = model_3)


if __name__ == "__main__":
    main()