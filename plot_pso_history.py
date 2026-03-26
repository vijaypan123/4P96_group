import matplotlib.pyplot as plt

# Replace this with your actual PSO history from the final run
history = [0.5330, 0.5464, 0.5464]

iterations = list(range(len(history)))

plt.figure(figsize=(8, 5))
plt.plot(iterations, history, marker='o')
plt.xlabel("PSO Iteration")
plt.ylabel("Best Validation Accuracy")
plt.title("PSO Convergence History")
plt.grid(True)
plt.tight_layout()
plt.savefig("pso_convergence.png", dpi=300)
plt.show()

print("Saved plot to: pso_convergence.png")