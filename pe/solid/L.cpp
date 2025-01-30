#include <iostream>

class Bird {
public:
    virtual void makeSound() = 0;
};

class FlyingBird : public Bird {
public:
    virtual void fly() = 0;
};

class Eagle : public FlyingBird {
public:
    void fly() override {
        std::cout << "Eagle is flying." << std::endl;
    }
    
    void makeSound() override {
        std::cout << "Eagle chirps." << std::endl;
    }
};

class Penguin : public Bird {
public:
    void makeSound() override {
        std::cout << "Penguin honks." << std::endl;
    }
};

void makeBirdFly(FlyingBird& bird) {
    bird.fly();
}

void makeBirdSound(Bird& bird) {
    bird.makeSound();
}

int main() {
    Eagle eagle;
    Penguin penguin;

    makeBirdFly(eagle);  // Works fine
    makeBirdSound(eagle);  // Works fine
    makeBirdSound(penguin);  // Works fine

    return 0;
}
