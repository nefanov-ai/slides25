# Constructor Variants in C++

In C++, constructors are special member functions that are automatically called when an object of a class is created. They are used to initialize objects. C++ supports several types of constructors, each serving different purposes. Below are the common constructor variants with examples:

---

## 1. **Default Constructor**
A default constructor is a constructor that takes no arguments. If no constructor is defined, the compiler automatically generates a default constructor.

```cpp
class MyClass {
public:
    MyClass() {
        std::cout << "Default Constructor Called!" << std::endl;
    }
};

int main() {
    MyClass obj; // Default constructor is called
    return 0;
}
```

---

## 2. **Parameterized Constructor**
A parameterized constructor accepts parameters to initialize an object with specific values.

```cpp
class Point {
    int x, y;
public:
    Point(int a, int b) : x(a), y(b) {
        std::cout << "Parameterized Constructor Called! (" << x << ", " << y << ")" << std::endl;
    }
};

int main() {
    Point p(10, 20); // Parameterized constructor is called
    return 0;
}
```

---

## 3. **Copy Constructor**
A copy constructor initializes an object using another object of the same class. It is used to create a copy of an existing object.

```cpp
class MyClass {
public:
    int value;
    MyClass(int v) : value(v) {}
    MyClass(const MyClass& other) : value(other.value) {
        std::cout << "Copy Constructor Called! Value: " << value << std::endl;
    }
};

int main() {
    MyClass obj1(10);
    MyClass obj2 = obj1; // Copy constructor is called
    return 0;
}
```

---

## 4. **Move Constructor (C++11 and later)**
A move constructor transfers resources from one object to another, typically used with dynamic memory or other resources to avoid unnecessary copying.

```cpp
class MyClass {
public:
    int* data;
    MyClass(int size) {
        data = new int[size];
        std::cout << "Constructor Called!" << std::endl;
    }
    MyClass(MyClass&& other) noexcept : data(other.data) {
        other.data = nullptr; // Transfer ownership
        std::cout << "Move Constructor Called!" << std::endl;
    }
    ~MyClass() {
        delete[] data;
        std::cout << "Destructor Called!" << std::endl;
    }
};

int main() {
    MyClass obj1(10);
    MyClass obj2 = std::move(obj1); // Move constructor is called
    return 0;
}
```

---

## 5. **Delegating Constructor (C++11 and later)**
A delegating constructor allows one constructor to call another constructor in the same class to avoid code duplication.

```cpp
class MyClass {
public:
    int a, b;
    MyClass() : MyClass(0, 0) { // Delegates to the parameterized constructor
        std::cout << "Delegating Constructor Called!" << std::endl;
    }
    MyClass(int x, int y) : a(x), b(y) {
        std::cout << "Parameterized Constructor Called! (" << a << ", " << b << ")" << std::endl;
    }
};

int main() {
    MyClass obj; // Delegating constructor is called
    return 0;
}
```

---

## 6. **Explicit Constructor**
An explicit constructor prevents implicit conversions or copy-initialization. It ensures that the constructor is only called explicitly.

```cpp
class MyClass {
public:
    int value;
    explicit MyClass(int v) : value(v) {
        std::cout << "Explicit Constructor Called! Value: " << value << std::endl;
    }
};

int main() {
    // MyClass obj = 10; // Error: Implicit conversion not allowed
    MyClass obj(10); // Explicit constructor is called
    return 0;
}
```

---

## 7. **Constructor with Default Arguments**
A constructor can have default arguments, allowing it to be called with fewer arguments than its parameters.

```cpp
class MyClass {
public:
    int a, b;
    MyClass(int x = 0, int y = 0) : a(x), b(y) {
        std::cout << "Constructor with Default Args Called! (" << a << ", " << b << ")" << std::endl;
    }
};

int main() {
    MyClass obj1;       // Uses default arguments (0, 0)
    MyClass obj2(10);   // Uses default for y (10, 0)
    MyClass obj3(10, 20); // Uses provided arguments (10, 20)
    return 0;
}
```

---

## Summary
| Constructor Type        | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| Default Constructor     | No parameters; initializes object with default values.                     |
| Parameterized Constructor | Takes parameters; initializes object with specific values.                 |
| Copy Constructor        | Initializes object using another object of the same class.                 |
| Move Constructor        | Transfers resources from one object to another (C++11 and later).          |
| Delegating Constructor  | Calls another constructor in the same class (C++11 and later).             |
| Explicit Constructor    | Prevents implicit conversions or copy-initialization.                      |
| Default Arguments       | Allows constructor to be called with fewer arguments than its parameters.  |

Each constructor variant serves a unique purpose and can be used depending on the requirements of your program.

---
Initializer list: 
Order of Initialization: Members are initialized in the order they are declared in the class, not the order they appear in the initializer list.
Mandatory for const and References: const members and reference members must be initialized using an initializer list.
Efficiency: Using an initializer list avoids default initialization followed by assignment, which can be inefficient for complex objects.
---

In C++, a **destructor** is a special member function that is automatically called when an object goes out of scope or is explicitly deleted. It is used to release resources (e.g., memory, file handles, etc.) that the object may have acquired during its lifetime. The destructor has the same name as the class but is prefixed with a tilde (`~`).

Below are the destructors for each of the constructor types mentioned earlier:


Use Case	Description
Basic Initialization	Initialize member variables directly.
const Members	Mandatory for const members.
Reference Members	Mandatory for reference members.
Member Objects Without Default CTOR	Required for objects without default constructors.
Order of Initialization	Members are initialized in the order they are declared in the class.
Efficiency	Avoids unnecessary default initialization and assignment.
Base Class Initialization	Used to call the base class constructor in inheritance.

---

## 1. **Destructor for Default Constructor**
```cpp
class MyClass {
public:
    MyClass() {
        std::cout << "Default Constructor Called!" << std::endl;
    }
    ~MyClass() {
        std::cout << "Destructor Called!" << std::endl;
    }
};

int main() {
    MyClass obj; // Default constructor is called
    // Destructor is called automatically when `obj` goes out of scope
    return 0;
}
```

---

## 2. **Destructor for Parameterized Constructor**
```cpp
class Point {
    int x, y;
public:
    Point(int a, int b) : x(a), y(b) {
        std::cout << "Parameterized Constructor Called! (" << x << ", " << y << ")" << std::endl;
    }
    ~Point() {
        std::cout << "Destructor Called! (" << x << ", " << y << ")" << std::endl;
    }
};

int main() {
    Point p(10, 20); // Parameterized constructor is called
    // Destructor is called automatically when `p` goes out of scope
    return 0;
}
```

---

## 3. **Destructor for Copy Constructor**
```cpp
class MyClass {
public:
    int value;
    MyClass(int v) : value(v) {}
    MyClass(const MyClass& other) : value(other.value) {
        std::cout << "Copy Constructor Called! Value: " << value << std::endl;
    }
    ~MyClass() {
        std::cout << "Destructor Called! Value: " << value << std::endl;
    }
};

int main() {
    MyClass obj1(10);
    MyClass obj2 = obj1; // Copy constructor is called
    // Destructors are called automatically when `obj1` and `obj2` go out of scope
    return 0;
}
```

---

## 4. **Destructor for Move Constructor**
```cpp
class MyClass {
public:
    int* data;
    MyClass(int size) {
        data = new int[size];
        std::cout << "Constructor Called!" << std::endl;
    }
    MyClass(MyClass&& other) noexcept : data(other.data) {
        other.data = nullptr; // Transfer ownership
        std::cout << "Move Constructor Called!" << std::endl;
    }
    ~MyClass() {
        delete[] data; // Release dynamically allocated memory
        std::cout << "Destructor Called!" << std::endl;
    }
};

int main() {
    MyClass obj1(10);
    MyClass obj2 = std::move(obj1); // Move constructor is called
    // Destructors are called automatically when `obj1` and `obj2` go out of scope
    return 0;
}
```

---

## 5. **Destructor for Delegating Constructor**
```cpp
class MyClass {
public:
    int a, b;
    MyClass() : MyClass(0, 0) { // Delegates to the parameterized constructor
        std::cout << "Delegating Constructor Called!" << std::endl;
    }
    MyClass(int x, int y) : a(x), b(y) {
        std::cout << "Parameterized Constructor Called! (" << a << ", " << b << ")" << std::endl;
    }
    ~MyClass() {
        std::cout << "Destructor Called! (" << a << ", " << b << ")" << std::endl;
    }
};

int main() {
    MyClass obj; // Delegating constructor is called
    // Destructor is called automatically when `obj` goes out of scope
    return 0;
}
```

---

## 6. **Destructor for Explicit Constructor**
```cpp
class MyClass {
public:
    int value;
    explicit MyClass(int v) : value(v) {
        std::cout << "Explicit Constructor Called! Value: " << value << std::endl;
    }
    ~MyClass() {
        std::cout << "Destructor Called! Value: " << value << std::endl;
    }
};

int main() {
    MyClass obj(10); // Explicit constructor is called
    // Destructor is called automatically when `obj` goes out of scope
    return 0;
}
```

---

## 7. **Destructor for Constructor with Default Arguments**
```cpp
class MyClass {
public:
    int a, b;
    MyClass(int x = 0, int y = 0) : a(x), b(y) {
        std::cout << "Constructor with Default Args Called! (" << a << ", " << b << ")" << std::endl;
    }
    ~MyClass() {
        std::cout << "Destructor Called! (" << a << ", " << b << ")" << std::endl;
    }
};

int main() {
    MyClass obj1;       // Uses default arguments (0, 0)
    MyClass obj2(10);   // Uses default for y (10, 0)
    MyClass obj3(10, 20); // Uses provided arguments (10, 20)
    // Destructors are called automatically when `obj1`, `obj2`, and `obj3` go out of scope
    return 0;
}
```

---

## Summary
The destructor is responsible for cleaning up resources when an object is destroyed. It is automatically called when:
1. An object goes out of scope.
2. An object is explicitly deleted (for dynamically allocated objects).
3. The program terminates.

Each class should have a destructor if it manages resources like dynamic memory, file handles, or network connections. If no destructor is defined, the compiler generates a default destructor, which may not properly release resources.

---

# Virtual Destructor in C++

In C++, a **virtual destructor** is needed when you have a base class with derived classes, and you intend to delete objects of the derived class through a pointer to the base class. Without a virtual destructor, only the base class destructor will be called, leading to **resource leaks** or **undefined behavior** if the derived class has dynamically allocated resources.

---

## Why Virtual Destructor is Needed

When you delete an object through a pointer to its base class, the destructor of the derived class will only be called if the base class destructor is **virtual**. If the base class destructor is not virtual, the behavior is undefined, and only the base class destructor will be called, leaving the derived class's resources un-cleaned.

---

## Key Points
1. **Polymorphic Base Classes**: If a class is intended to be used as a base class (i.e., it has at least one virtual function), its destructor should be virtual.
2. **Resource Management**: Ensures proper cleanup of resources in derived classes.
3. **Prevent Undefined Behavior**: Avoids undefined behavior when deleting derived objects through base class pointers.

---

## Example Without Virtual Destructor

```cpp
#include <iostream>
class Base {
public:
    Base() { std::cout << "Base Constructor\n"; }
    ~Base() { std::cout << "Base Destructor\n"; } // Non-virtual destructor
};

class Derived : public Base {
public:
    Derived() { std::cout << "Derived Constructor\n"; }
    ~Derived() { std::cout << "Derived Destructor\n"; }
};

int main() {
    Base* ptr = new Derived(); // Base pointer to Derived object
    delete ptr; // Only Base destructor is called
    return 0;
}
```

**Output:**
```
Base Constructor
Derived Constructor
Base Destructor
```

**Problem**: The `Derived` destructor is not called, leading to potential resource leaks.

---

## Example With Virtual Destructor

```cpp
#include <iostream>
class Base {
public:
    Base() { std::cout << "Base Constructor\n"; }
    virtual ~Base() { std::cout << "Base Destructor\n"; } // Virtual destructor
};

class Derived : public Base {
public:
    Derived() { std::cout << "Derived Constructor\n"; }
    ~Derived() { std::cout << "Derived Destructor\n"; }
};

int main() {
    Base* ptr = new Derived(); // Base pointer to Derived object
    delete ptr; // Both Derived and Base destructors are called
    return 0;
}
```

**Output:**
```
Base Constructor
Derived Constructor
Derived Destructor
Base Destructor
```

**Solution**: The `Derived` destructor is called first, followed by the `Base` destructor, ensuring proper cleanup.

---

## When to Use a Virtual Destructor

1. **Polymorphic Base Classes**: If a class is intended to be inherited and used polymorphically (i.e., through base class pointers), its destructor should be virtual.
2. **Dynamic Resource Management**: If the derived class manages resources (e.g., dynamic memory, file handles, etc.), a virtual destructor ensures proper cleanup.
3. **Prevent Undefined Behavior**: If you delete a derived object through a base class pointer, a virtual destructor ensures the correct destructor sequence.

---

## When Not to Use a Virtual Destructor

1. **Non-Polymorphic Classes**: If a class is not intended to be used as a base class or is not used polymorphically, a virtual destructor is unnecessary.
2. **Performance Considerations**: Virtual functions introduce a small overhead due to the vtable. If performance is critical and the class is not polymorphic, avoid virtual destructors.

---

## Example with Multiple Levels of Inheritance

```cpp
#include <iostream>
class Base {
public:
    Base() { std::cout << "Base Constructor\n"; }
    virtual ~Base() { std::cout << "Base Destructor\n"; } // Virtual destructor
};

class Derived1 : public Base {
public:
    Derived1() { std::cout << "Derived1 Constructor\n"; }
    ~Derived1() { std::cout << "Derived1 Destructor\n"; }
};

class Derived2 : public Derived1 {
public:
    Derived2() { std::cout << "Derived2 Constructor\n"; }
    ~Derived2() { std::cout << "Derived2 Destructor\n"; }
};

int main() {
    Base* ptr = new Derived2(); // Base pointer to Derived2 object
    delete ptr; // Destructors are called in reverse order
    return 0;
}
```

**Output:**
```
Base Constructor
Derived1 Constructor
Derived2 Constructor
Derived2 Destructor
Derived1 Destructor
Base Destructor
```

---

## Summary

| Scenario                              | Virtual Destructor Needed? |
|---------------------------------------|----------------------------|
| Base class with derived classes       | Yes                        |
| Polymorphic base class                | Yes                        |
| Derived class manages resources       | Yes                        |
| Non-polymorphic base class            | No                         |
| Performance-critical code             | No (unless polymorphic)    |

### Rule of Thumb
- If a class has **any virtual functions**, it should also have a **virtual destructor**.
- If a class is **not intended for inheritance**, mark it as `final` and avoid virtual destructors.

By using virtual destructors, you ensure that your program behaves correctly and avoids resource leaks when working with polymorphic objects.

---

