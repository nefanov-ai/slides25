#!/bin/bash

# Update package list and upgrade existing packages
sudo apt update && sudo apt upgrade -y

# Setup sources.list
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Setup keys
sudo apt install curl -y
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Update package list
sudo apt update

# Install ROS Noetic
sudo apt install ros-noetic-desktop-full -y

# Setup environment
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install dependencies for building packages
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y

# Initialize rosdep
sudo rosdep init
rosdep update

# Create a catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make

echo "ROS Noetic has been installed successfully!"
