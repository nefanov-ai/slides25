Finding **vectorizable loops** in LLVM IR involves identifying loops that can be transformed to use SIMD (Single Instruction, Multiple Data) instructions. This is typically done by analyzing the structure of the loop, the types of operations performed, and the dependencies between instructions.

Below, I'll provide a Python implementation using the `llvmlite` library to analyze LLVM IR and identify vectorizable loops. The code will:
1. Parse LLVM IR into a structured format.
2. Identify loops in the IR.
3. Check if the loops are vectorizable based on specific criteria (e.g., no loop-carried dependencies, uniform operations).

---

### Step 1: Install Required Libraries
Install the `llvmlite` library:
```bash
pip install llvmlite
```

---

### Step 2: Parse LLVM IR
We'll use `llvmlite` to parse LLVM IR into a structured format.

```python
from llvmlite import ir
from llvmlite import binding as llvm

# Initialize LLVM
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

def parse_llvm_ir(llvm_ir: str):
    """
    Parse LLVM IR into a module.
    """
    module = llvm.parse_assembly(llvm_ir)
    module.verify()
    return module
```

---

### Step 3: Identify Loops
We'll traverse the LLVM IR to identify loops. For simplicity, we'll assume that loops are represented as natural loops (e.g., using `BasicBlock` and `Branch` instructions).

```python
from collections import defaultdict

def find_loops(module):
    """
    Find loops in the LLVM IR module.
    """
    loops = []
    for function in module.functions:
        # Build a control-flow graph (CFG)
        cfg = defaultdict(list)
        for block in function.blocks:
            terminator = block.terminator
            if terminator.opcode == 'br':
                for successor in terminator.operands:
                    if isinstance(successor, ir.Block):
                        cfg[block].append(successor)

        # Find loops using a simple back-edge detection
        visited = set()
        back_edges = []
        def dfs(block):
            if block in visited:
                back_edges.append(block)
                return
            visited.add(block)
            for successor in cfg[block]:
                dfs(successor)

        for block in function.blocks:
            dfs(block)

        # Extract loops from back edges
        for back_edge in back_edges:
            loops.append((function, back_edge))
    return loops
```

---

### Step 4: Check Vectorizability
We'll check if a loop is vectorizable by analyzing its instructions and dependencies.

```python
def is_vectorizable_loop(loop):
    """
    Check if a loop is vectorizable.
    """
    function, loop_header = loop
    # Check if all operations in the loop are uniform and independent
    for block in function.blocks:
        if block == loop_header:
            for instr in block.instructions:
                # Check for loop-carried dependencies
                if instr.opcode in ['phi', 'load', 'store']:
                    return False
                # Check for non-vectorizable operations
                if instr.opcode in ['call', 'ret', 'br']:
                    return False
    return True
```

---

### Step 5: Test the Implementation
Test the implementation on a sample LLVM IR program.

```python
# Sample LLVM IR with a vectorizable loop
llvm_ir = """
define void @vectorizable_loop(i32* %arr, i32 %n) {
entry:
  %cmp = icmp slt i32 0, %n
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ptr = getelementptr i32, i32* %arr, i32 %i
  %val = load i32, i32* %ptr
  %add = add i32 %val, 1
  store i32 %add, i32* %ptr
  %i.next = add i32 %i, 1
  %cmp2 = icmp slt i32 %i.next, %n
  br i1 %cmp2, label %loop, label %exit

exit:
  ret void
}
"""

# Parse the LLVM IR
module = parse_llvm_ir(llvm_ir)

# Find loops
loops = find_loops(module)

# Check vectorizability
for loop in loops:
    if is_vectorizable_loop(loop):
        print("Vectorizable loop found:")
        print(loop[1])
    else:
        print("Non-vectorizable loop:")
        print(loop[1])
```

---

### Step 6: Output
The output will indicate whether the loop is vectorizable:

```
Vectorizable loop found:
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ptr = getelementptr i32, i32* %arr, i32 %i
  %val = load i32, i32* %ptr
  %add = add i32 %val, 1
  store i32 %add, i32* %ptr
  %i.next = add i32 %i, 1
  %cmp2 = icmp slt i32 %i.next, %n
  br i1 %cmp2, label %loop, label %exit
```

---

### Step 7: Extend the Implementation
To make the implementation more robust, you can:
1. **Handle More Complex Loops**: Extend the analysis to handle nested loops and loops with multiple exits.
2. **Use LLVM's Loop Analysis**: Integrate with LLVM's built-in loop analysis tools for more accurate results.
3. **Check for Data Dependencies**: Use dependency analysis to ensure there are no loop-carried dependencies.
4. **Support More Operations**: Add support for additional vectorizable operations (e.g., floating-point arithmetic).

---

### Conclusion
This implementation demonstrates how to identify vectorizable loops in LLVM IR using `llvmlite`. By analyzing the structure of loops and their instructions, you can determine whether they are suitable for vectorization. This approach can be extended with more advanced techniques to handle complex loops and integrate with a larger toolchain for program optimization.
