### **FEAC: Federated Learning with Error Accumulation and Compensation for Signed Gradient Over-the-Air Aggregation**
Federated learning (FL) has recently gained significant attention in machine learning due to its decentralised approach enabling parallelised computation across multiple edge devices while aggregating model updates at a parameter server.
However, FL often suffers from limited bandwidth resulting in communication latency and overhead during wireless transmission of model updates between devices and the parameter server.
To address this limitation, sign-based gradient compression has been proposed which effectively reduces communication costs by transmitting only the sign of each gradient component.
Nevertheless, such extreme compression can lead to significant information loss and makes the system vulnerable due to transmission errors particularly in wireless environments where noise and fading
are predominant. These challenges can cause model divergence during training or slow down convergence compared to full-precision transmission.

This paper proposes an error feedback method that accumulates the difference between the original and extremely compressed gradients and compensates the error during the next update.
This approach compensates information loss due to sign compression and improves both convergence speed and final model accuracy.
Furthermore, we consider both uplink and downlink channels to be imperfect reflecting real-world wireless communication scenarios.
We provide a theoretical convergence analysis showing the effect of error feedback on convergence rate and validate this with experiments on a real dataset.
The results demonstrate that the proposed method not only reduces communication overhead but also preserves model robustness in FL over bandwidth-constrained and noisy wireless networks.
