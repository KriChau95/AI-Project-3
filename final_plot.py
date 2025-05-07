import matplotlib.pyplot as plt

# Read data from file
avg_remaining_moves_map = {}
with open("avg_remaining_moves.txt", "r") as f:
    for line in f:
        if ":" in line:
            key_str, value_str = line.strip().split(":")
            key = int(key_str.strip())
            value = float(value_str.strip())
            avg_remaining_moves_map[key] = value

# Prepare data for plotting
L_sizes = sorted(avg_remaining_moves_map.keys())
avg_moves = [avg_remaining_moves_map[L] for L in L_sizes]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(L_sizes, avg_moves, marker='o', linestyle='-', color='blue')
plt.title("Average Remaining Moves vs L Size (From File)")
plt.xlabel("|L| (Distinct L Sizes)")
plt.ylabel("Average Remaining Moves at Midpoint Occurrence")
plt.grid(True)
plt.tight_layout()
plt.show()
