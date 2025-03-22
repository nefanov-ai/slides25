import compiler_gym

def compile_and_measure(benchmark: str, optimization_flag: str):
    # Initialize the CompilerGym environment
    env = compiler_gym.make(
        "llvm-v0",  # Use the LLVM environment
        benchmark=benchmark,  # Specify the benchmark
    )

    # Set the optimization flag using commandline
    env.commandline(optimization_flag)

    # Compile the benchmark
    env.reset()

    # Get the runtime and code size
    runtime = env.observation["Runtime"]
    code_size = env.observation["IrInstructionCountOz"]

    print(f"Benchmark: {benchmark}")
    print(f"Optimization Flag: {optimization_flag}")
    print(f"Runtime: {runtime}")
    print(f"Code Size: {code_size}")

    # Close the environment
    env.close()

# Example usage
if __name__ == "__main__":
    benchmark = "cbench-v1/crc32"  # Specify the benchmark
    optimization_flag = "-Oz"  # Specify the optimization flag
    compile_and_measure(benchmark, optimization_flag)
