import jax
import flax
import pennylane as qml

print("JAX version:", jax.__version__)
print("Flax version:", flax.__version__)
print("PennyLane version:", qml.__version__)
print(jax.devices())