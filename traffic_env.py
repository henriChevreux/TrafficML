import numpy as np
import gym
from gym import spaces

class TrafficGridEnv(gym.Env):
    """
    A simple traffic grid environment with traffic lights controlling intersections.
    
    State:
        - Queue length at each approach to each intersection (4 approaches per intersection)
        - Current phase of each traffic light
    
    Action:
        - Change phase of traffic lights (0: NS green, 1: EW green)
    
    Reward:
        - Number of cars that passed through intersections during the timestep
    """
    
    def __init__(self, grid_size=2):
        super(TrafficGridEnv, self).__init__()
        
        self.grid_size = grid_size
        self.num_intersections = grid_size * grid_size
        
        # Define action and observation space
        # Actions: each traffic light can be either NS green (0) or EW green (1)
        self.action_space = spaces.Discrete(2**self.num_intersections)
        
        # Observations: queue length at each approach (4 per intersection) + current phase
        self.observation_space = spaces.Box(
            low=0,
            high=100,  # Maximum queue length
            shape=(self.num_intersections * 4 + self.num_intersections,),
            dtype=np.int32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        # Initialize queue lengths for each approach at each intersection
        self.queue_lengths = np.random.randint(0, 10, size=(self.num_intersections, 4))
        
        # Initialize traffic light phases (0: NS green, 1: EW green)
        self.traffic_light_phases = np.zeros(self.num_intersections, dtype=np.int32)
        
        # Arrival rates for each approach at each intersection
        self.arrival_rates = np.random.uniform(0.1, 0.5, size=(self.num_intersections, 4))
        
        # Maximum service rate when the light is green
        self.service_rate = 5
        
        return self._get_observation()
    
    def step(self, action):
        """Take a step in the environment."""
        # Convert the action (integer) to binary representation for each traffic light
        binary_action = format(action, f'0{self.num_intersections}b')
        
        for i in range(self.num_intersections):
            self.traffic_light_phases[i] = int(binary_action[i])
        
        # Process traffic flow
        cars_passed = self._process_traffic_flow()
        
        # Generate new arrivals
        self._generate_arrivals()
        
        # Calculate reward as the number of cars that passed
        reward = cars_passed
        
        # Check if simulation should end
        done = False  # In this simple version, we'll run indefinitely
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """Construct the observation from the current state."""
        queue_lengths_flat = self.queue_lengths.flatten()
        return np.concatenate([queue_lengths_flat, self.traffic_light_phases])
    
    def _process_traffic_flow(self):
        """Process traffic flow based on current traffic light phases."""
        total_cars_passed = 0
        
        for i in range(self.num_intersections):
            phase = self.traffic_light_phases[i]
            
            if phase == 0:  # NS green, EW red
                # Process North-South traffic (approaches 0 and 2)
                for approach in [0, 2]:
                    cars_to_pass = min(self.queue_lengths[i, approach], self.service_rate)
                    self.queue_lengths[i, approach] -= cars_to_pass
                    total_cars_passed += cars_to_pass
            else:  # EW green, NS red
                # Process East-West traffic (approaches 1 and 3)
                for approach in [1, 3]:
                    cars_to_pass = min(self.queue_lengths[i, approach], self.service_rate)
                    self.queue_lengths[i, approach] -= cars_to_pass
                    total_cars_passed += cars_to_pass
        
        return total_cars_passed
    
    def _generate_arrivals(self):
        """Generate new car arrivals at each approach."""
        arrivals = np.random.poisson(self.arrival_rates)
        self.queue_lengths += arrivals
        
        # Ensure queue lengths don't exceed maximum
        self.queue_lengths = np.clip(self.queue_lengths, 0, 100)
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            for i in range(self.num_intersections):
                phase = "NS Green" if self.traffic_light_phases[i] == 0 else "EW Green"
                print(f"Intersection {i}: {phase}")
                print(f"  Queue lengths: N={self.queue_lengths[i, 0]}, E={self.queue_lengths[i, 1]}, "
                      f"S={self.queue_lengths[i, 2]}, W={self.queue_lengths[i, 3]}")
        return 