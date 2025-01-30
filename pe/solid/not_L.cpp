#include <iostream>

class Bird {
public:
    virtual void fly() = 0;
};

class Eagle : public Bird {
public:
    void fly() override {
        std::cout << "Eagle is flying." << std::endl;
    }
};

class Penguin : public Bird {
public:
    void fly() override {
        throw std::runtime_error("Penguins cannot fly.");
    }
};

void makeBirdFly(Bird& bird) {
    bird.fly();
}

int main() {
    Eagle eagle;
    Penguin penguin;

    makeBirdFly(eagle);  // Works fine
    try {
        makeBirdFly(penguin);  // Throws an exception
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}

//This example does not perfectly adhere to the LSP because Penguin cannot fulfill the contract of Bird (flying). 
//A better design would be to create a separate class hierarchy for birds that can fly and those that cannot:
