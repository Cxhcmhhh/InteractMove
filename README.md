# InteractMove

This repository is the official Pytorch implementation for the paper **InteractMove: Text-Controlled Human-Object Interaction Generation in 3D Scenes with Movable Objects**.

In this paper, we propose a novel task of text-controlled human-object interaction generation in 3D scenes with movable objects. 
We construct the InteractMove dataset for Movable Human-Object Interaction in 3D Scenes by aligning existing human-object interaction data with scene contexts, featuring three key characteristics: 1) scenes containing multiple movable objects with text-controlled interaction specifications (including same-category distractors requiring spatial and 3D scene context understanding), 2) diverse object types and sizes with varied interaction patterns (one-hand, two-hand, etc.), and 3) physically plausible object manipulation trajectories. 
We first use 3D visual grounding models to identify the interaction object. Then, we propose a hand-object joint affordance learning to predict contact regions for different hand joints and object parts, enabling accurate grasping and manipulation of diverse objects. Finally, we optimize interactions with local-scene modeling and collision avoidance constraints, ensuring physically plausible motions and avoiding collisions between objects and the scene.

Our paper was accepted by ACM-MM 2025. 

The code will be made public soon.
