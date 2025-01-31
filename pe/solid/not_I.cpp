#include <iostream>

// Fat interface
class IWorker {
public:
    virtual void work() = 0;
    virtual void eat() = 0;
};

class Worker : public IWorker {
public:
    void work() override {
        std::cout << "Worker is working." << std::endl;
    }
    
    void eat() override {
        std::cout << "Worker is eating." << std::endl;
    }
};

class Robot : public IWorker {
public:
    void work() override {
        std::cout << "Robot is working." << std::endl;
    }
    
    // Dummy implementation for eat() since robots don't eat
    void eat() override {
        std::cout << "Robot does not eat." << std::endl;
    }
};

int main() {
    Worker worker;
    worker.work();
    worker.eat();
    
    Robot robot;
    robot.work();
    robot.eat();  // Forced to implement eat() even though robots don't eat
    
    return 0;
}
Good Example (Following ISP)


cpp
#include <iostream>

// Segregated interfaces
class IWorkable {
public:
    virtual void work() = 0;
};

class IFeedable {
public:
    virtual void eat() = 0;
};

class Worker : public IWorkable, public IFeedable {
public:
    void work() override {
        std::cout << "Worker is working." << std::endl;
    }
    
    void eat() override {
        std::cout << "Worker is eating." << std::endl;
    }
};

class Robot : public IWorkable {
public:
    void work() override {
        std::cout << "Robot is working." << std::endl;
    }
};

int main() {
    Worker worker;
    worker.work();
    worker.eat();
    
    Robot robot;
    robot.work();  // No need to implement eat()
    
    return 0;
}
