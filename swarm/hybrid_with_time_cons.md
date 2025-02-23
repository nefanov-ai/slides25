To incorporate a **long-term constraint on the arrival time** for all drones into the constraints, we need to define a new constraint that ensures all drones reach their target positions within a specified time limit. This constraint can be added to the existing constraints (collision avoidance) and enforced using a **Lagrangian multiplier**.

Below, I provide the updated **theorem** and **algorithm modifications** to include this new constraint.

---

### **Updated Theorem: Mean and Maximal Probability of Collision with Arrival Time Constraint**

#### **Assumptions**
1. **Drone Dynamics**:
   - Each drone \( i \) moves in a 3D space with position \( \mathbf{p}_i(t) \in \mathbb{R}^3 \) and velocity \( \mathbf{v}_i(t) \in \mathbb{R}^3 \) at time \( t \).
   - The velocity is controlled by the action \( \mathbf{a}_i(t) \in \mathbb{R}^3 \).

2. **Collision Constraints**:
   - A collision occurs if the Euclidean distance between two drones \( \|\mathbf{p}_i(t) - \mathbf{p}_j(t)\| < d_{\text{min}} \) or between a drone and an obstacle \( \|\mathbf{p}_i(t) - \mathbf{o}_k(t)\| < d_{\text{min}} \), where \( d_{\text{min}} > 0 \) is the minimum safe distance.

3. **Arrival Time Constraint**:
   - All drones must reach their target positions within a maximum allowed time \( T_{\text{max}}} \).
   - Let \( T_i \) be the time taken by drone \( i \) to reach its target. The constraint is:
     \[
     \max_{i \in \{1, 2, \dots, N\}} T_i \leq T_{\text{max}}}.
     \]

4. **Constraint Enforcement**:
   - The constraints are enforced using Lagrangian multipliers:
     - \( \lambda_{\text{short}}} \) for the short-term collision constraint.
     - \( \lambda_{\text{long}}} \) for the long-term collision constraint.
     - \( \lambda_{\text{arrival}}} \) for the arrival time constraint.

5. **Probability of Collision**:
   - Let \( P_{\text{collision}}}(t) \) denote the probability of collision at time \( t \).
   - The **mean probability of collision** over an episode is:
     \[
     \mathbb{E}[P_{\text{collision}}}] = \frac{1}{T} \sum_{t=1}^T P_{\text{collision}}}(t).
     \]
   - The **maximal probability of collision** over an episode is:
     \[
     \max P_{\text{collision}}} = \max_{t \in \{1, 2, \dots, T\}} P_{\text{collision}}}(t).
     \]

---

### **Theorem Statement**

Let:
- \( \epsilon_{\text{short}}} > 0 \) be the threshold for the short-term collision constraint.
- \( \epsilon_{\text{long}}} > 0 \) be the threshold for the long-term collision constraint.
- \( \epsilon_{\text{arrival}}} > 0 \) be the threshold for the arrival time constraint.
- \( d_{\text{min}}} > 0 \) be the minimum safe distance.
- \( T_{\text{max}}} > 0 \) be the maximum allowed arrival time.
- \( T > 0 \) be the episode length.

Then:
1. **Mean Probability of Collision**:
   \[
   \mathbb{E}[P_{\text{collision}}}] \leq \frac{\epsilon_{\text{long}}}}{T \cdot d_{\text{min}}^2}.
   \]

2. **Maximal Probability of Collision**:
   \[
   \max P_{\text{collision}}} \leq \frac{\epsilon_{\text{short}}}}{d_{\text{min}}^2}.
   \]

3. **Arrival Time Constraint**:
   \[
   \max_{i \in \{1, 2, \dots, N\}} T_i \leq T_{\text{max}}} + \frac{\epsilon_{\text{arrival}}}}{\alpha},
   \]
   where \( \alpha > 0 \) is a scaling factor that depends on the dynamics of the drones and the environment.

---

### **Algorithm Modifications**

To incorporate the **arrival time constraint**, we need to:
1. Define a cost function for the arrival time constraint.
2. Add a Lagrangian multiplier \( \lambda_{\text{arrival}}} \) to enforce the constraint.
3. Update the policy optimization objective to include the arrival time constraint.

#### **Cost Function for Arrival Time Constraint**
Define the arrival time cost as:
\[
C_{\text{arrival}}} = \max_{i \in \{1, 2, \dots, N\}} \left( T_i - T_{\text{max}}} \right),
\]
where \( T_i \) is the time taken by drone \( i \) to reach its target.

#### **Updated Policy Optimization Objective**
The policy optimization objective now includes the arrival time constraint:
\[
L^{CPO}(\theta_i) = L^{CLIP}(\theta_i) - \lambda_{\text{short}}} \mathbb{E}[C_{\text{short}}}] - \lambda_{\text{long}}} \mathbb{E}[C_{\text{long}}}] - \lambda_{\text{arrival}}} \mathbb{E}[C_{\text{arrival}}}],
\]
where:
- \( L^{CLIP}(\theta_i) \) is the PPO clipped surrogate loss for policy \( \pi_i \).
- \( C_{\text{short}}} \) is the short-term collision cost.
- \( C_{\text{long}}} \) is the long-term collision cost.
- \( C_{\text{arrival}}} \) is the arrival time cost.

#### **Update Lagrangian Multipliers**
The Lagrangian multipliers are updated as follows:
1. **Short-Term Collision Constraint**:
   \[
   \lambda_{\text{short}}} \leftarrow \max \left( 0, \lambda_{\text{short}}} + \eta_{\text{short}}} \cdot \mathbb{E}[C_{\text{short}}}] \right).
   \]

2. **Long-Term Collision Constraint**:
   \[
   \lambda_{\text{long}}} \leftarrow \max \left( 0, \lambda_{\text{long}}} + \eta_{\text{long}}} \cdot \left( \mathbb{E}[C_{\text{long}}}] - \epsilon_{\text{long}}} \right) \right).
   \]

3. **Arrival Time Constraint**:
   \[
   \lambda_{\text{arrival}}} \leftarrow \max \left( 0, \lambda_{\text{arrival}}} + \eta_{\text{arrival}}} \cdot \left( \mathbb{E}[C_{\text{arrival}}}] - \epsilon_{\text{arrival}}} \right) \right).
   \]

---

### **Interpretation**
1. **Arrival Time Constraint**:
   - Ensures that all drones reach their targets within \( T_{\text{max}} + \frac{\epsilon_{\text{arrival}}}{\alpha} \).
   - The term \( \frac{\epsilon_{\text{arrival}}}}{\alpha} \) accounts for the flexibility in enforcing the constraint.

2. **Trade-Offs**:
   - Tightening the arrival time constraint (reducing \( T_{\text{max}}} \) or \( \epsilon_{\text{arrival}}} \)) may increase the collision probability.
   - Balancing the arrival time constraint with collision avoidance is crucial for safe and efficient operation.

---

### **Example**

Suppose:
- \( T_{\text{max}}} = 100 \) (maximum allowed arrival time is 100 time steps).
- \( \epsilon_{\text{arrival}}} = 10 \) (flexibility in arrival time constraint).
- \( \alpha = 1 \) (scaling factor).

Then:
\[
\max_{i \in \{1, 2, \dots, N\}} T_i \leq 100 + \frac{10}{1} = 110.
\]

This means all drones must reach their targets within **110 time steps**.

---

This updated theorem and algorithm now include the **arrival time constraint** alongside collision avoidance. Let me know if you need further clarification or additional details!
