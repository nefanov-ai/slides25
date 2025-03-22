import compiler_gym

# Create an environment for the cbench/crc32 benchmark
env = compiler_gym.make(
    "llvm-v0",  # Use the LLVM environment
    benchmark="cbench-v1/crc32",  # Specify the benchmark
)

# Reset the environment to the initial state
env.reset()

# Define a list of compiler flags (actions) to apply
# These are just examples; the actual flags depend on the compiler and optimization goals
compiler_flags = [
    "-O1",  # Optimization level 1
    "-fvectorize",  # Enable vectorization
    "-floop-unroll",  # Enable loop unrolling
    "-finline",  # Enable function inlining
]

# Apply each compiler flag (action) to the environment
for flag in compiler_flags:
    # Convert the flag to an action index
    action = env.action_space.from_string(flag)
    # Step the environment with the action
    observation, reward, done, info = env.step(action)

    # Print the results of the step
    print(f"Applied flag: {flag}")
    print(f"Observation: {observation}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")

# Close the environment
env.close()
