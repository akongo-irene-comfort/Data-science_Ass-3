"""
Traffic Environment using SUMO simulator
Gymnasium-compatible interface for DQN training
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import traci
    import sumolib
except ImportError:
    print("WARNING: SUMO not installed. Install with: pip install traci sumolib")
    traci = None
    sumolib = None


class TrafficEnv(gym.Env):
    """
    Custom Traffic Environment for RL training

    Observation Space: Continuous vector of traffic state
    Action Space: Discrete (8 traffic signal phases)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sumo_cfg="sumo/grid.sumocfg", gui=False, max_steps=3600):
        super(TrafficEnv, self).__init__()

        self.sumo_cfg = sumo_cfg
        self.use_gui = gui
        self.max_steps = max_steps

        # State dimensions: 8 lanes * 3 features (count, speed, density) + time
        state_dim = 8 * 3 + 1  # 25 dimensions

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # 8 discrete actions (traffic signal phases)
        self.action_space = spaces.Discrete(8)

        self.current_step = 0
        self.total_waiting_time = 0
        self.total_queue_length = 0
        self.vehicles_completed = 0

        # SUMO connection
        self.sumo_cmd = None
        self.connection = None

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        # Close existing connection
        if self.connection is not None:
            try:
                traci.close()
            except:
                pass

        # Start SUMO
        if self.use_gui:
            sumo_binary = "sumo-gui"
        else:
            sumo_binary = "sumo"

        sumo_cmd = [
            sumo_binary,
            "-c",
            self.sumo_cfg,
            "--start",
            "--quit-on-end",
            "--no-warnings",
            "--random",
        ]

        if traci:
            traci.start(sumo_cmd)
            self.connection = traci

        self.current_step = 0
        self.total_waiting_time = 0
        self.total_queue_length = 0
        self.vehicles_completed = 0

        # Get initial state
        state = self._get_state()
        info = self._get_info()

        return state, info

    def step(self, action):
        """Execute action and return next state"""

        # Apply action (change traffic light phase)
        self._apply_action(action)

        # Run simulation for action duration (10 seconds)
        action_duration = 10
        for _ in range(action_duration):
            if self.connection:
                traci.simulationStep()
            self.current_step += 1

            # Update metrics
            self.total_waiting_time += self._get_waiting_time()
            self.total_queue_length += self._get_queue_length()

        # Get new state
        state = self._get_state()
        reward = self._compute_reward()
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = self._get_info()

        return state, reward, terminated, truncated, info

    def _get_state(self):
        """Get current traffic state observation"""
        if not self.connection:
            # Return dummy state if SUMO not connected
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        state = []

        # Get state for each lane
        lane_ids = traci.lane.getIDList()[:8]  # First 8 lanes

        for lane_id in lane_ids:
            # Vehicle count
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
            state.append(vehicle_count)

            # Average speed
            avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            state.append(avg_speed)

            # Density (vehicles per meter)
            length = traci.lane.getLength(lane_id)
            density = vehicle_count / length if length > 0 else 0
            state.append(density)

        # Time of day (normalized 0-1)
        time_of_day = (self.current_step % 3600) / 3600.0
        state.append(time_of_day)

        return np.array(state, dtype=np.float32)

    def _apply_action(self, action):
        """Apply traffic light action"""
        if not self.connection:
            return

        # Map action to traffic light phase
        traffic_light_id = "J1"  # Assuming intersection ID

        phase_mapping = {
            0: 0,  # NS Green
            1: 1,  # NS Yellow
            2: 2,  # EW Green
            3: 3,  # EW Yellow
            4: 4,  # Left turn
            5: 5,  # Left yellow
            6: 6,  # All red
            7: 7,  # Pedestrian
        }

        try:
            phase = phase_mapping.get(action, 0)
            traci.trafficlight.setPhase(traffic_light_id, phase)
        except:
            pass

    def _compute_reward(self):
        """Compute reward based on multiple objectives"""

        # Get current metrics
        waiting_time = self._get_waiting_time()
        queue_length = self._get_queue_length()
        throughput = self._get_throughput()

        # Multi-objective reward function
        # R = -α·W - β·Q + γ·T
        alpha = 1.0  # Waiting time weight
        beta = 0.5  # Queue length weight
        gamma = 0.8  # Throughput weight

        reward = -alpha * waiting_time - beta * queue_length + gamma * throughput

        return reward

    def _get_waiting_time(self):
        """Get total waiting time of all vehicles"""
        if not self.connection:
            return 0

        waiting_time = 0
        for veh_id in traci.vehicle.getIDList():
            waiting_time += traci.vehicle.getWaitingTime(veh_id)

        return waiting_time

    def _get_queue_length(self):
        """Get total queue length"""
        if not self.connection:
            return 0

        queue_length = 0
        for lane_id in traci.lane.getIDList()[:8]:
            queue_length += traci.lane.getLastStepHaltingNumber(lane_id)

        return queue_length

    def _get_throughput(self):
        """Get number of vehicles that completed their trip"""
        if not self.connection:
            return 0

        # Count arrived vehicles in this step
        arrived = traci.simulation.getArrivedNumber()
        self.vehicles_completed += arrived

        return arrived

    def _get_info(self):
        """Get additional information"""
        avg_waiting_time = self.total_waiting_time / max(self.current_step, 1)
        avg_queue_length = self.total_queue_length / max(self.current_step, 1)

        return {
            "step": self.current_step,
            "avg_waiting_time": avg_waiting_time,
            "avg_queue_length": avg_queue_length,
            "throughput": self.vehicles_completed,
            "total_waiting_time": self.total_waiting_time,
        }

    def close(self):
        """Close the environment"""
        if self.connection:
            try:
                traci.close()
            except:
                pass
        self.connection = None

    def render(self):
        """Render the environment (handled by SUMO GUI)"""
        pass
