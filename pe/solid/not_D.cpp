// D principle of solid C++ violation
#include <iostream>

class LightBulb {
public:
    void TurnOn() {
        std::cout << "Light bulb on..." << std::endl;
    }
    
    void TurnOff() {
        std::cout << "Light bulb off..." << std::endl;
    }
};

class ElectricPowerSwitch {
public:
    ElectricPowerSwitch(LightBulb& light_bulb) : light_bulb_(light_bulb), on_(false) {}
    
    void press() {
        if (!on_) {
            light_bulb_.TurnOn();
            on_ = true;
        } else {
            light_bulb_.TurnOff();
            on_ = false;
        }
    }
    
private:
    LightBulb& light_bulb_;
    bool on_;
};

int main() {
    LightBulb light_bulb;
    ElectricPowerSwitch switch_(light_bulb);
    switch_.press();  // Turns on the light bulb
    switch_.press();  // Turns off the light bulb
    
    return 0;
}

/*

+---------------+
|  LightBulb   |
+---------------+
|  TurnOn()    |
|  TurnOff()   |
+---------------+
         |
         | Composition
         v
+-------------------+
| ElectricPowerSwitch|
+-------------------+
|  press()         |
|  LightBulb&      |
+-------------------+
*/  
