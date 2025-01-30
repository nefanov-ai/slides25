#include <iostream>
#include <string>

// Class responsible for managing user data
class User {
private:
    std::string name;
    int age;

public:
    User(const std::string& name, int age) : name(name), age(age) {}

    std::string getName() const { return name; }
    int getAge() const { return age; }
};

// Class responsible for saving user data to a file
class UserSaver {
public:
    void saveToFile(const User& user, const std::string& filename) {
        // Simulate saving user data to a file
        std::cout << "Saving user " << user.getName() << " to file: " << filename << std::endl;
    }
};
