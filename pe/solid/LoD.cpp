class A {
public:
    void doSomething() {
        // A only interacts with B
        b.doSomething();
    }

private:
    B b;
};

class B {
public:
    void doSomething() {
        // B interacts with C
        c.doSomethingElse();
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
