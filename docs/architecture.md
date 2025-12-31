## AM Architecture

AM is composed of four primary subsystems:

1. **Abstraction Autoencoder**  
   Learns compact latent representations of sensory inputs.

2. **World Model**  
   Predicts transitions in latent space, enabling imagination-based planning.

3. **Policy Network**  
   Maps latent states to actions.

4. **Experience Consolidation**  
   Maintains continual learning via replay and consolidation cycles.

The agent can simulate possible futures using its learned world model, allowing it to evaluate actions before execution.
