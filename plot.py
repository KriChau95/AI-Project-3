import matplotlib.pyplot as plt

# Read the data from the text file
L_sizes = []
avg_moves = []

with open("localization_results.txt", "r") as f:
    for line in f:
        parts = line.strip().split(",")
        L = int(parts[0].split("=")[1])
        moves = float(parts[1].split("=")[1])
        L_sizes.append(L)
        avg_moves.append(moves)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(L_sizes, avg_moves, marker='o')
plt.xlabel("Initial |L| (Number of Possible Locations)")
plt.ylabel("Average Number of Moves Needed")
plt.title("Localization Efficiency vs. Initial Uncertainty")
plt.gca().invert_xaxis()  # Optional: so it goes from max |L| to min
plt.grid(True)
plt.tight_layout()
plt.show()
