/*
+-------------------+
| SwitchableDevice |
+-------------------+
|  TurnOn()        |
|  TurnOff()       |
+-------------------+
         |
         | Implements
         v
+---------------+
|  LightBulb     |
+---------------+
|  TurnOn()      |
|  TurnOff()     |
+---------------+
         |
         | Composition
         v
+-------------------+
| ElectricPowerSwitch|
+-------------------+
|  press()         |
|  SwitchableDevice&|
+-------------------+

Dependency Inverted Implementation:

- ElectricPowerSwitch depends on the abstract SwitchableDevice.
- LightBulb implements SwitchableDevice.
The composition relationship is shown by the arrow from ElectricPowerSwitch to SwitchableDevice.
Benefits
- Decoupling: High-level modules (ElectricPowerSwitch) are decoupled from low-level modules (LightBulb).
- Reusability: ElectricPowerSwitch becomes more reusable since it depends on an abstraction (SwitchableDevice) rather than a specific implementation (LightBulb).
- Flexibility: New devices can be easily added by implementing the SwitchableDevice interface.

*/

#include <iostream>

// Abstract interface for devices
class SwitchableDevice {
public:
    virtual void TurnOn() = 0;
    virtual void TurnOff() = 0;
};

class LightBulb : public SwitchableDevice {
public:
    void TurnOn() override {
        std::cout << "Light bulb on..." << std::endl;
    }
    
    void TurnOff() override {
        std::cout << "Light bulb off..." << std::endl;
    }
};

class ElectricPowerSwitch {
public:
    ElectricPowerSwitch(SwitchableDevice& device) : device_(device), on_(false) {}
    
    void press() {
        if (!on_) {
            device_.TurnOn();
            on_ = true;
        } else {
            device_.TurnOff();
            on_ = false;
        }
    }
    
private:
    SwitchableDevice& device_;
    bool on_;
};

int main() {
    LightBulb light_bulb;
    ElectricPowerSwitch switch_(light_bulb);
    switch_.press();  // Turns on the light bulb
    switch_.press();  // Turns off the light bulb
    
    return 0;
}
