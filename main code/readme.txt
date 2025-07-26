Dynamic Multi-Target Fusion for Efficient Audio-Visual Navigation
Based on the Habitat-Sim framework for Audio-Visual Navigation (AVN) using Reinforcement Learning, this system is trained and evaluated with the PPO algorithm.


simple\_agents.py
Provides five minimal baseline agents (random, forward-only, forward+turn, biased-random-forward, and goal-oriented GoalFollower), useful for quickly validating the environment or serving as performance lower bounds for benchmarks.

base\_trainer.py
Defines the base `Trainer` classes `BaseTrainer` / `BaseRLTrainer`, standardizing the interfaces for training, evaluation, and checkpointing, and providing generic logic for multi-checkpoint evaluation.

baseline\_registry.py
Extends the Habitat registry by offering decorators `@register_trainer` and `@register_env`, allowing centralized management of training algorithms and environment implementations, promoting modularity.

benchmark.py
Implements the `Benchmark` class: given a config, it can run any `habitat.Agent` over multiple episodes in `NavRLEnv`, returning metrics such as SPL and Success.

env\_utils.py
Builds parallelized `VectorEnv` instances based on config (supports `SyncVectorEnv`, `VectorEnv`, and `ThreadedVectorEnv`), partitions data by scenes to improve throughput.

environments.py
Registers and defines `NavRLEnv`: wraps the native Habitat environment, implements RL semantics such as reward, done, and info, and supports audio-visual multimodal input.

rollout\_storage.py
`RolloutStorage`: stores observations, actions, log-probs, values, rewards, and masks over a `num_steps × num_envs` buffer for PPO. Supports GAE computation and recurrent batch sampling.

sync\_vector\_env.py
Custom synchronous `VectorEnv` implementation (multi-process → multi-threaded `WorkerEnv`), works with `env_utils`, and supports pause/resume and debugging individual environments.

tensorboard\_utils.py
`TensorboardWriter` wrapper: auto-creates writers and supports writing numpy frame sequences as videos directly to TensorBoard.

utils.py
General utility functions: tensor/observation batch processing, checkpoint polling, video generation (with audio), top-down trajectory visualization, network initialization, etc.

audio\_cnn.py
`AudioCNN`: a 2D CNN for processing spectrograms (e.g., audiogoal/spectrogram) and outputting fixed-dimensional embeddings for the policy network.

modeling\_resnet.py
Predefined ResNet-V2 backbone (with GroupNorm and Weight Standardization), used for feature extraction in Hybrid-ViT models.

rnn\_state\_encoder.py
`RNNStateEncoder`: a wrapper for GRU/LSTM that supports variable-length sequences and mask resetting, used for final state aggregation in the policy network.

transformer.py
Audio-Visual Fusion Transformer:
`Embeddings/Embedding`: ViT patch embedding with optional ResNet hybrid backbone;
`Transformer`: two encoders (one for vision, one for audio) and one decoder with cross-attention fusion, outputs a 1024-dimensional vector;
Includes helper modules like `TensorTransformer` and `LinearTransform`.

visual\_cnn.py
Same content as `encoder.py` (different filename), maintained for backward compatibility; outputs CNN feature maps instead of embeddings.

policy.py
Policy network:
`PointNavBaselineNet` combines `VisualCNN`, `AudioCNN`, `Transformer`, and `RNNStateEncoder`;
`Policy` wrapper includes a categorical action head and critic, exposing `act`, `evaluate_actions`, and `get_value` interfaces.

ppo.py
Implements PPO loss components: GAE, clipped surrogate loss, value clipping, entropy regularization, Adam optimizer, and gradient clipping.

ppo\_trainer.py
`PPOTrainer`: inherits from `BaseRLTrainer`, implements the full training loop (rollout → GAE → PPO update → checkpoint/evaluation), supports video visualization and metric logging.

run.py
Entry script: parses command-line arguments (e.g., train/eval mode, config path, checkpoint path), launches the appropriate trainer via the registry, and starts the experiment.


Execution Pipeline:
`run.py` → `ppo_trainer.py` → `policy.py` (+ `transformer` / `audio_cnn` / `visual_cnn`) → `env_utils.py` (+ `environments`) → `base_trainer.py` / `ppo.py` → Habitat-Sim & SoundSpaces environment.

