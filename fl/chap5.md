Plan:

### **Week 5: Semantic Analysis**

**Objective:**  
This week focuses on the semantic analysis phase of the compiler, where the compiler ensures that the program is semantically correct. Students will learn about type checking, symbol tables, and scope management, and will implement a basic semantic analyzer.

---

### **Topics:**

1. **Role of the Semantic Analyzer:**
   - Understanding the purpose of semantic analysis in the compilation process.
   - Differences between syntax and semantics.
   - Common semantic errors (e.g., type mismatches, undeclared variables).

2. **Type Checking:**
   - Static vs. dynamic typing.
   - Type systems: strong vs. weak typing, type inference.
   - Type compatibility and coercion.
   - Implementing type checking rules for expressions, assignments, and function calls.

3. **Symbol Tables:**
   - Purpose and structure of symbol tables.
   - Storing and retrieving information about variables, functions, and types.
   - Handling nested scopes and block structures.

4. **Scope Management:**
   - Lexical (static) scope vs. dynamic scope.
   - Implementing scope rules for variables and functions.
   - Handling shadowing and redeclaration of variables.

---

### **Lab: Implementing a Symbol Table and Type Checker**

**Lab Objectives:**
- Implement a symbol table to store information about variables and functions.
- Implement a basic type checker for a small language.

**Lab Tasks:**
1. **Symbol Table Implementation:**
   - Define a data structure for the symbol table (e.g., hash table, tree).
   - Implement functions to insert, lookup, and delete entries in the symbol table.
   - Handle nested scopes by maintaining a stack of symbol tables.

2. **Type Checker Implementation:**
   - Define a set of type-checking rules for expressions, assignments, and function calls.
   - Implement a function to traverse the Abstract Syntax Tree (AST) and perform type checking.
   - Report semantic errors (e.g., type mismatches, undeclared variables).

**Lab Deliverables:**
- A working symbol table implementation.
- A basic type checker that can handle a small subset of a programming language (e.g., arithmetic expressions, variable declarations).

---

### **Assignment: Extend the Semantic Analyzer**

**Assignment Objectives:**
- Extend the semantic analyzer to handle more complex language features.
- Gain experience in implementing semantic analysis for a real-world programming language.

**Assignment Tasks:**
1. **Extend the Symbol Table:**
   - Add support for function declarations and parameter lists.
   - Handle nested scopes and block structures.

2. **Extend the Type Checker:**
   - Implement type checking for control flow statements (e.g., if-else, loops).
   - Add support for user-defined types (e.g., structs, classes).
   - Handle type coercion and implicit type conversions.

3. **Error Reporting:**
   - Improve error reporting to provide meaningful messages for semantic errors.
   - Ensure that the semantic analyzer can recover from errors and continue checking the rest of the program.

**Assignment Deliverables:**
- A report describing the design and implementation of the extended semantic analyzer.
- Source code for the extended semantic analyzer, including the symbol table and type checker.
- Test cases demonstrating the semantic analyzer's ability to handle various language features and report errors.

---

### **Reading Materials:**
- **Textbook:** "Compilers: Principles, Techniques, and Tools" by Aho, Lam, Sethi, and Ullman (Chapter 6: Intermediate-Code Generation)
- **Online Resources:**
  - [LLVM Documentation on Semantic Analysis](https://llvm.org/docs/)
  - [ANTLR Documentation on Semantic Predicates](https://github.com/antlr/antlr4/blob/master/doc/predicates.md)

---

### **Assessment:**
- **Lab Work:** 10% of the course grade
- **Assignment:** 20% of the course grade

---

### **Additional Notes:**
- Encourage students to use version control (e.g., Git) for their lab and assignment work.
- Provide sample code and test cases to help students get started with their implementations.
- Offer office hours or additional help sessions for students who need assistance with the lab or assignment.

This week's plan provides a solid foundation in semantic analysis, with hands-on experience in implementing a symbol table and type checker, which are crucial components of any compiler.

Text:

### **Common Semantic Errors and the Role of the Semantic Analyzer**

In the compilation process, **semantic analysis** is the phase where the compiler ensures that the program is not only syntactically correct but also meaningful according to the rules of the programming language. While syntax analysis checks whether the program is well-formed (e.g., proper use of grammar rules), semantic analysis focuses on the **meaning** of the program. This involves verifying that variables are declared before use, types are compatible, and operations are valid.

In this lesson, we’ll explore **common semantic errors** and how the **semantic analyzer** detects and handles them.

---

### **What Are Semantic Errors?**

Semantic errors occur when the code is syntactically correct but violates the rules of the language. These errors are often subtle and can lead to unexpected behavior during program execution. Here are some common examples:

1. **Type Mismatch Errors:**
   - Occurs when an operation is performed on incompatible types.
   - Example: Adding an integer to a string (`5 + "hello"`).

2. **Undeclared or Undefined Variables:**
   - Occurs when a variable is used without being declared.
   - Example: Using `x` without a prior declaration like `int x;`.

3. **Redeclaration of Variables:**
   - Occurs when a variable is declared more than once in the same scope.
   - Example: `int x; int x;` in the same block.

4. **Scope Violations:**
   - Occurs when a variable is used outside its declared scope.
   - Example: Using a local variable outside the function where it was declared.

5. **Function Call Errors:**
   - Occurs when a function is called with the wrong number or types of arguments.
   - Example: Calling a function `add(int a, int b)` with `add(5)` or `add(5, "hello")`.

6. **Incompatible Assignments:**
   - Occurs when a value is assigned to a variable of an incompatible type.
   - Example: Assigning a string to an integer variable (`int x = "hello";`).

7. **Missing Return Statements:**
   - Occurs when a function is expected to return a value but does not.
   - Example: A function declared as `int foo()` does not return an integer.

---

### **Role of the Semantic Analyzer**

The **semantic analyzer** is responsible for detecting and reporting these errors. It works closely with the **symbol table** and **type system** to enforce the language’s semantic rules. Here’s how it operates:

1. **Type Checking:**
   - The semantic analyzer ensures that all operations and assignments are type-safe.
   - It checks that operands in expressions have compatible types and that function arguments match the expected parameter types.

2. **Symbol Table Management:**
   - The semantic analyzer uses the **symbol table** to track information about variables, functions, and types.
   - It ensures that variables are declared before use and that there are no duplicate declarations in the same scope.

3. **Scope Resolution:**
   - The semantic analyzer enforces scope rules, ensuring that variables are used within their valid scope.
   - It handles nested scopes, such as blocks within functions, and resolves variable references correctly.

4. **Error Reporting:**
   - When a semantic error is detected, the semantic analyzer generates meaningful error messages to help the programmer debug the issue.
   - It often tries to recover from errors to continue analyzing the rest of the program.

---

### **How the Semantic Analyzer Works**

The semantic analyzer typically operates on the **Abstract Syntax Tree (AST)** generated by the parser. Here’s a step-by-step overview of its workflow:

1. **Traverse the AST:**
   - The semantic analyzer walks through the AST, visiting each node to perform semantic checks.

2. **Type Inference and Checking:**
   - For each expression, the analyzer infers the type and checks it against the expected type.
   - Example: In the expression `a + b`, it ensures that `a` and `b` are of compatible types (e.g., both integers).

3. **Symbol Table Lookups:**
   - For each variable or function reference, the analyzer looks up the symbol table to verify that it has been declared and is in scope.

4. **Scope Management:**
   - When entering a new scope (e.g., a function or block), the analyzer pushes a new symbol table onto the stack.
   - When exiting the scope, it pops the symbol table from the stack.

5. **Error Handling:**
   - If a semantic error is detected, the analyzer reports the error and may attempt to recover by making assumptions (e.g., defaulting to a specific type).

---

### **Example: Detecting a Semantic Error**

Consider the following code snippet:
```c
int main() {
    int x = 5;
    y = x + "hello";  // Semantic error: adding an integer to a string
    return 0;
}
```

Here’s how the semantic analyzer would handle this:
1. It detects that `y` is undeclared and reports an error.
2. It detects that `x + "hello"` involves incompatible types (integer and string) and reports a type mismatch error.

---

### **Why Semantic Analysis Matters**

Semantic analysis is crucial because it ensures that the program behaves as intended. Without it, many errors would go unnoticed until runtime, leading to crashes, incorrect results, or security vulnerabilities. By catching these errors early, the semantic analyzer helps programmers write safer and more reliable code.

---

### **Key Takeaways**
- Semantic errors occur when the code violates the rules of the language, even if it is syntactically correct.
- The semantic analyzer enforces type safety, scope rules, and other semantic constraints.
- It uses the symbol table and type system to track and verify information about variables, functions, and types.
- Effective error reporting and recovery are essential features of a good semantic analyzer.

By understanding these concepts, you’ll be well-prepared to implement your own semantic analyzer in the lab and assignment this week!


### **Type Checking in Semantic Analysis**

Type checking is one of the most critical tasks performed by the **semantic analyzer** during the compilation process. It ensures that the operations and expressions in a program are **type-safe**, meaning that the types of operands are compatible with the operations being performed. Without proper type checking, programs may exhibit undefined behavior, crashes, or incorrect results at runtime.

In this lesson, we’ll explore the concept of **type checking**, how it works, and its role in ensuring the correctness of a program.

---

### **What is Type Checking?**

Type checking is the process of verifying that the types of values, variables, and expressions in a program are used correctly according to the rules of the programming language. It ensures that:
- Operands in expressions have compatible types.
- Function calls use the correct number and types of arguments.
- Assignments are made to variables of compatible types.
- Return statements match the expected return type of a function.

Type checking can be performed at two stages:
1. **Static Type Checking:** Done at compile time. The compiler checks types before the program runs.
2. **Dynamic Type Checking:** Done at runtime. The program checks types during execution.

Most modern programming languages (e.g., Java, C++, Python) use a combination of static and dynamic type checking, with an emphasis on static type checking to catch errors early.

---

### **Why is Type Checking Important?**

Type checking plays a crucial role in ensuring program correctness and reliability. Here’s why it matters:
1. **Prevents Type Errors:**
   - Catches errors like adding a string to an integer or calling a function with the wrong arguments.
2. **Improves Code Quality:**
   - Helps programmers write safer and more predictable code.
3. **Enhances Performance:**
   - Static type checking allows the compiler to optimize code more effectively.
4. **Provides Better Debugging:**
   - Type errors are caught early, making it easier to debug and fix issues.

---

### **How Type Checking Works**

The type checker operates on the **Abstract Syntax Tree (AST)** generated by the parser. It traverses the AST and verifies that the types of operands and expressions are consistent with the language’s type rules. Here’s a step-by-step overview of the process:

1. **Type Inference:**
   - The type checker determines the type of each expression based on the types of its operands and the operation being performed.
   - Example: In the expression `a + b`, if `a` and `b` are integers, the result is also an integer.

2. **Type Compatibility:**
   - The type checker ensures that the types of operands are compatible with the operation.
   - Example: Addition (`+`) is allowed for numeric types (e.g., `int`, `float`) but not for a string and an integer.

3. **Type Conversion (Coercion):**
   - In some cases, the type checker may implicitly convert one type to another to make them compatible.
   - Example: Adding an integer and a float may involve converting the integer to a float.

4. **Function Call Validation:**
   - The type checker ensures that function calls use the correct number and types of arguments.
   - Example: A function `int add(int a, int b)` must be called with two integer arguments.

5. **Assignment Validation:**
   - The type checker ensures that the value being assigned to a variable is compatible with the variable’s declared type.
   - Example: Assigning a string to an integer variable is invalid.

6. **Error Reporting:**
   - If a type error is detected, the type checker generates a meaningful error message to help the programmer fix the issue.

---

### **Examples of Type Checking**

Let’s look at some examples to understand how type checking works in practice.

#### Example 1: Type Mismatch in Expressions
```c
int x = 5;
string y = "hello";
int z = x + y;  // Error: Cannot add an integer and a string
```
- The type checker detects that `x` is an integer and `y` is a string.
- It reports an error because the `+` operation is not defined for these types.

#### Example 2: Function Call with Incorrect Arguments
```c
int add(int a, int b) {
    return a + b;
}

int result = add(5, "hello");  // Error: Second argument must be an integer
```
- The type checker verifies that the arguments passed to `add` match the parameter types.
- It reports an error because `"hello"` is not an integer.

#### Example 3: Incompatible Assignment
```c
int x;
x = "hello";  // Error: Cannot assign a string to an integer variable
```
- The type checker ensures that the value being assigned to `x` is compatible with its declared type (`int`).
- It reports an error because `"hello"` is a string.

---

### **Type Systems**

The type checker relies on the **type system** of the programming language, which defines the rules for types and their interactions. There are several types of type systems:

1. **Static vs. Dynamic Typing:**
   - **Static Typing:** Types are checked at compile time (e.g., Java, C++).
   - **Dynamic Typing:** Types are checked at runtime (e.g., Python, JavaScript).

2. **Strong vs. Weak Typing:**
   - **Strong Typing:** Enforces strict type rules and disallows implicit type conversions (e.g., Python).
   - **Weak Typing:** Allows implicit type conversions and is more lenient with type rules (e.g., C).

3. **Type Inference:**
   - Some languages (e.g., Haskell, TypeScript) can infer the types of variables and expressions automatically, reducing the need for explicit type annotations.

---

### **Implementing a Type Checker**

In the lab and assignment this week, you will implement a basic type checker. Here’s what you’ll need to do:
1. **Define Type Rules:**
   - Specify the types supported by your language (e.g., `int`, `float`, `string`).
   - Define the rules for type compatibility and operations.

2. **Traverse the AST:**
   - Write a function to traverse the AST and perform type checking on each node.

3. **Handle Errors:**
   - Report type errors with meaningful messages.
   - Optionally, recover from errors to continue checking the rest of the program.

4. **Test Your Implementation:**
   - Create test cases to verify that your type checker works correctly.

---

### **Key Takeaways**
- Type checking ensures that operations and expressions in a program are type-safe.
- It prevents type errors, improves code quality, and enhances performance.
- The type checker operates on the AST and enforces the language’s type rules.
- Static type checking catches errors at compile time, while dynamic type checking catches errors at runtime.

By understanding and implementing type checking, you’ll gain a deeper appreciation for how compilers ensure the correctness and reliability of programs. Let’s get started on building your type checker in the lab!
