import gym
from gym import spaces
import numpy as np
import pymunk

class DoublePendulumEnv(gym.Env):
    def __init__(self, reward_type="shaped"):
        super().__init__()

        self.reward_type = reward_type

        # Observation: 6 values
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Action: force [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.dt = 1 / 60.0
        self.force_scale = 5000

        self.space = pymunk.Space()
        self.space.gravity = (0, -900)

        self.reset()

    def reset(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, -900)

        # CART
        mass = 5
        size = (50, 20)
        moment = pymunk.moment_for_box(mass, size)

        self.cart_body = pymunk.Body(mass, moment)
        self.cart_body.position = (300, 100)

        cart_shape = pymunk.Poly.create_box(self.cart_body, size)
        cart_shape.friction = 1.0

        # TRACK (constraint)
        static_body = self.space.static_body
        groove = pymunk.GrooveJoint(
            static_body,
            self.cart_body,
            (100, 100),
            (500, 100),
            (0, 0)
        )

        # POLE 1
        pole_mass = 1
        pole_length = 100

        self.pole1_body = pymunk.Body(
            pole_mass,
            pymunk.moment_for_segment(pole_mass, (0, 0), (0, pole_length), 5)
        )
        self.pole1_body.position = (300, 150)

        pole1_shape = pymunk.Segment(
            self.pole1_body, (0, 0), (0, pole_length), 5
        )

        # JOINT: cart → pole1
        joint1 = pymunk.PivotJoint(
            self.cart_body, self.pole1_body, (300, 120)
        )

        # POLE 2
        self.pole2_body = pymunk.Body(
            pole_mass,
            pymunk.moment_for_segment(pole_mass, (0, 0), (0, pole_length), 5)
        )
        self.pole2_body.position = (300, 250)

        pole2_shape = pymunk.Segment(
            self.pole2_body, (0, 0), (0, pole_length), 5
        )

        # JOINT: pole1 → pole2
        joint2 = pymunk.PivotJoint(
            self.pole1_body, self.pole2_body, (300, 250)
        )

        self.space.add(
            self.cart_body, cart_shape,
            self.pole1_body, pole1_shape,
            self.pole2_body, pole2_shape,
            groove, joint1, joint2
        )

        return self._get_obs()

    def _get_obs(self):
        return np.array([
            self.cart_body.position.x,
            self.cart_body.velocity.x,
            self.pole1_body.angle,
            self.pole1_body.angular_velocity,
            self.pole2_body.angle,
            self.pole2_body.angular_velocity,
        ], dtype=np.float32)

    def step(self, action):
        force = float(action[0]) * self.force_scale

        # Apply force to cart
        self.cart_body.apply_force_at_local_point((force, 0), (0, 0))

        # Step simulation
        self.space.step(self.dt)

        obs = self._get_obs()

        theta1 = self.pole1_body.angle
        theta2 = self.pole2_body.angle

        # BASELINE reward
        base_reward = np.cos(theta1) + np.cos(theta2)

        if self.reward_type == "baseline":
            reward = base_reward
        else:
            # SHAPED reward
            reward = base_reward
            reward -= abs(self.cart_body.position.x - 300) * 0.001
            reward -= abs(self.pole1_body.angular_velocity) * 0.01
            reward -= abs(self.pole2_body.angular_velocity) * 0.01
            reward -= (action[0] ** 2) * 0.001

        done = (
            abs(theta1) > np.pi / 2
            or abs(theta2) > np.pi / 2
        )

        return obs, reward, done, {}

    def render(self, mode="human"):
        pass