#include <iostream>
#include <vector>

// Base class for shapes
class Shape {
public:
    virtual double area() const = 0; // Pure virtual function
};

// Derived class for Circle
class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) : radius(r) {}

    double area() const override {
        return 3.14159 * radius * radius;
    }
};

// Derived class for Rectangle
class Rectangle : public Shape {
private:
    double width, height;

public:
    Rectangle(double w, double h) : width(w), height(h) {}

    double area() const override {
        return width * height;
    }
};

// Function to calculate total area of shapes
double totalArea(const std::vector<Shape*>& shapes) {
    double total = 0;
    for (const auto& shape : shapes) {
        total += shape->area();
    }
    return total;
}

int main() {
    Circle circle(5);
    Rectangle rectangle(4, 6);

    std::vector<Shape*> shapes = {&circle, &rectangle};
    std::cout << "Total area: " << totalArea(shapes) << std::endl;

    return 0;
}
