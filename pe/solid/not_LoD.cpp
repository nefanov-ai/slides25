class A {
public:
    void doSomething() {
        // Directly accessing a method of a deeply nested object
        b.getC().doSomethingElse();
    }

private:
    B b;
};

class B {
public:
    C& getC() {
        return c;
    }

private:
    C c;
};

class C {
public:
    void doSomethingElse() {
        // Some implementation
    }
};
