import compiler_gym
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import TimeLimit

# Initialize the CompilerGym environment
env = compiler_gym.make(
    "llvm-v0",
    benchmark="cbench-v1/crc32",
    observation_space="Runtime",
    reward_space="IrInstructionCountOz",
)

# Apply the -Oz optimization flag
env.reset()
env.apply(["-Oz"])

# Compile the benchmark
observation, reward, done, info = env.step(env.action_space.sample())

# Get the runtime and code size
runtime = env.observation["Runtime"]
code_size = env.observation["IrInstructionCountOz"]

print(f"Runtime: {runtime}")
print(f"Code Size: {code_size}")

# Close the environment
env.close()
