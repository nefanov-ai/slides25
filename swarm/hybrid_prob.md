The **maximal probability of collision** in the theorem above is given by the following formula:

\[
\max P_{\text{collision}}} \leq \frac{\epsilon_{\text{short}}}}{d_{\text{min}}^2},
\]

where:
- \( \max P_{\text{collision}}} \) is the **maximal probability of collision** at any time step during the episode.
- \( \epsilon_{\text{short}}} \) is the **short-term constraint threshold**, which bounds the immediate collision risk at any time step.
- \( d_{\text{min}}} \) is the **minimum safe distance** between drones or between a drone and an obstacle.

---

### **Explanation of the Formula**

1. **Maximal Probability of Collision**:
   - This represents the **worst-case probability of collision** at any single time step during the episode.
   - It is derived from the **short-term constraint**, which ensures that the immediate collision risk is bounded by \( \epsilon_{\text{short}}} \).

2. **Short-Term Constraint Threshold (\( \epsilon_{\text{short}}} \))**:
   - This is the maximum allowable immediate collision risk at any time step.
   - It is enforced by the Lagrangian multiplier \( \lambda_{\text{short}}} \) in the algorithm.

3. **Minimum Safe Distance (\( d_{\text{min}}} \))**:
   - This is the minimum distance that must be maintained between drones or between a drone and an obstacle to avoid collisions.
   - The probability of collision is inversely proportional to the square of \( d_{\text{min}}} \), as collisions are less likely when drones are farther apart.

---

### **Derivation of the Formula**

The formula is derived based on the following reasoning:

1. **Collision Probability and Distance**:
   - The probability of collision \( P_{\text{collision}}} \) is inversely proportional to the square of the distance between drones or between a drone and an obstacle. This is because the "collision zone" (area where collisions can occur) scales with the square of the distance in a 3D space.

2. **Short-Term Constraint**:
   - The short-term constraint ensures that the immediate collision risk at any time step is bounded by \( \epsilon_{\text{short}}} \).
   - Mathematically, this can be expressed as:
     \[
     P_{\text{collision}}}(t) \leq \epsilon_{\text{short}}} \quad \forall t \in \{1, 2, \dots, T\}.
     \]

3. **Scaling with Minimum Safe Distance**:
   - Since the collision probability scales inversely with \( d_{\text{min}}^2 \), we have:
     \[
     P_{\text{collision}}}(t) \propto \frac{1}{d_{\text{min}}^2}.
     \]
   - Combining this with the short-term constraint gives:
     \[
     \max P_{\text{collision}}} \leq \frac{\epsilon_{\text{short}}}}{d_{\text{min}}^2}.
     \]

---

### **Interpretation**

- The formula shows that the **maximal probability of collision** is directly proportional to the short-term constraint threshold \( \epsilon_{\text{short}}} \) and inversely proportional to the square of the minimum safe distance \( d_{\text{min}}} \).
- To reduce the maximal probability of collision:
  - Decrease \( \epsilon_{\text{short}}} \) (tighten the short-term constraint).
  - Increase \( d_{\text{min}}} \) (maintain a larger minimum safe distance).

---

### **Example**

Suppose:
- \( \epsilon_{\text{short}}} = 0.1 \) (the short-term constraint allows a maximum immediate collision risk of 10%).
- \( d_{\text{min}}} = 2.0 \) (the minimum safe distance is 2.0 units).

Then:
\[
\max P_{\text{collision}}} \leq \frac{0.1}{2.0^2} = \frac{0.1}{4} = 0.025.
\]

This means the **maximal probability of collision** at any time step is **2.5%**.

---

This formula provides a clear and quantitative relationship between the short-term constraint, the minimum safe distance, and the worst-case collision probability. Let me know if you need further clarification!
