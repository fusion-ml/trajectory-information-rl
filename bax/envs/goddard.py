import gym
import numpy as np

class Rocket(object):

    '''
        Expected parameters and methods of the rocket environment being simulated.

        V0/H0/M0:   Initial velocity/height/weight
        M1:         Final weight (set equal to dry mass if all fuel should be used)
        THRUST_MAX: Maximum possible force of thrust [N]
        GAMMA:      Fuel consumption [kg/N/s]
        DT:         Assumed time [s] between calls to step()

    '''

    V0 = H0 = M0 = M1 = THRUST_MAX = GAMMA = DT = None

    H_MAX_RENDER = None # Sets upper window bound for rendering

    def drag(self, v, h):
        raise NotImplementedError

    def g(self, h):
        raise NotImplementedError

class Default(Rocket):

    '''
        Models the surface-to-air missile (SA-2 Guideline) described in
        http://dcsl.gatech.edu/papers/jgcd92.pdf
        https://www.mcs.anl.gov/~more/cops/bcops/rocket.html

        The equations of motion is made dimensionless by scaling and choosing
        the model parameters in terms of initial height, mass and gravity.
    '''

    DT = 0.001

    V0 = 0.0
    H0 = M0 = G0 = 1.0

    H_MAX_RENDER = 1.015

    HC = 500.0
    MC = 0.7
    VC = 620.0

    M1 = MC * M0

    thrust_to_weight_ratio = 3.5
    THRUST_MAX = thrust_to_weight_ratio * G0 * M0
    DC = 0.5 * VC * M0 / G0
    GAMMA = 1.0 / (0.5*np.sqrt(G0*H0))

    def drag(self, v, h):
        return self.DC * v * abs(v) * np.exp(-self.HC*(h-self.H0)/self.H0)

    def g(self, h):
        return self.G0 * (self.H0/h)**2


class BACRocket(Rocket):

    '''
        Modified default for BAC problem
    '''

    DT = 0.01

    V0 = 0.0
    H0 = M0 = G0 = 1.0

    H_MAX_RENDER = 1.015

    HC = 500.0
    MC = 0.6
    VC = 620.0

    M1 = MC * M0

    thrust_to_weight_ratio = 3.5
    THRUST_MAX = thrust_to_weight_ratio * G0 * M0
    DC = 0.5 * VC * M0 / G0
    GAMMA = 1.0 / (0.5*np.sqrt(G0*H0))

    def drag(self, v, h):
        return self.DC * v * abs(v) * np.exp(-self.HC*(h-self.H0)/self.H0)

    def g(self, h):
        return self.G0 * (self.H0/h)**2


class SaturnV(Rocket):

    '''
        Throttled first stage burn of Saturn V rocket

        Specifications taken from:
        1. https://en.wikipedia.org/wiki/Saturn_V
        2. https://web.archive.org/web/20170313142729/http://www.braeunig.us/apollo/saturnV.htm

    '''

    DT = 0.5
    G = 9.81
    V0 = 0.0
    H0 = 0.0

    H_MAX_RENDER = 170e3

    # Vehicle mass + fuel: 2.97e6 kg
    # Thrust of first stage: 35.1e6 N
    M0 = 2.97e6
    THRUST_MAX = 35.1e6

    # First stage gross and dry weight: 2.29e6 kg, 130e3 kg
    # => Saturn V gross weight immediately before first stage separation
    M1 = M0 - (2.29e6-130e3) # = 810e3 kg

    # Specific pulse: 263 s
    GAMMA = 1.0/(G*263.0) # = 387e-6 kg/N/s

    D = 0.35 * 0.27 * 113.0 / 2.0

    def drag(self, v, h):
        return self.D * v * abs(v)

    def g(self, h):
        return self.G

class GoddardEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, rocket=BACRocket()):
        super(GoddardEnv, self).__init__()

        self._r = rocket
        self.viewer = None
        self.horizon = 30

        self.U_INDEX = 0

        # changed from original to be [-1, 1], for consistency. we denormalize in the step function
        self.action_space = gym.spaces.Box(
            low   = np.array([-1.0]),
            high  = np.array([1.0]),
            shape = (1,),
            dtype = np.float
        )

        self.V_INDEX, self.H_INDEX, self.M_INDEX = 0, 1, 2
        self.observation_space_norm = gym.spaces.Box(
            low   = np.array([-0.15, 1, self._r.M1]),
            high  = np.array([0.15, 1.02, self._r.M0]),
            dtype = np.float
        )
        self.observation_space = gym.spaces.Box(
            low   = -np.ones_like(self.observation_space_norm.low),
            high  = np.ones_like(self.observation_space_norm.high),
            dtype = np.float
        )

        self.reset()

    def extras_labels(self):
        return ['action', 'thrust', 'drag', 'gravity']

    def step(self, action):
        v, h, m = self._state

        is_tank_empty = (m <= self._r.M1)

        # action is normalized to be between -1 and 1 for consistency
        a = -1.0 if is_tank_empty else action[self.U_INDEX]
        # this gets fixed here to be between 0 and 1
        a += 1
        a /= 2
        # should now be fixed
        thrust = self._r.THRUST_MAX*a

        self._thrust_last = thrust

        drag = self._r.drag(v,h)
        g = self._r.g(h)

        # Forward Euler
        self._state = (
            0.0 if h==self._r.H0 and v!=0.0 else (v + self._r.DT * ((thrust-drag)/m - g)),
            max(h + self._r.DT * v, self._r.H0),
            max(m - self._r.DT * self._r.GAMMA * thrust, self._r.M1)
        )
        new_m = self._state[2]

        self._h_max = max(self._h_max, self._state[self.H_INDEX])

        is_done = bool(
            is_tank_empty and self._state[self.V_INDEX] < 0 and self._h_max > self._r.H0
        )

        if is_done:
            reward = self._h_max - self._r.H0
        else:
            reward = 0.0

        extras = dict(zip(self.extras_labels(), [action[self.U_INDEX], thrust, drag, g]))

        return self._observation(), reward, is_done, extras

    def maximum_altitude(self):
        return self._h_max

    def normalize(self, unnorm):
        norm = (unnorm - self.observation_space_norm.low) / (self.observation_space_norm.high -
                                                           self.observation_space_norm.low)
        norm *= 2
        norm -= 1
        return norm

    def unnormalize(self, norm):
        unnorm = norm + 1
        unnorm /= 2
        unnorm = (unnorm * (self.observation_space_norm.high - self.observation_space_norm.low)) + \
                  self.observation_space_norm.low
        return unnorm

    def _observation(self):
        unnorm = np.array(self._state)
        return self.normalize(unnorm)

    def reset(self, obs=None):
        self._state = (self._r.V0, self._r.H0, self._r.M0)
        if obs is not None:
            unnorm = self.unnormalize(obs)
            self._state = tuple(unnorm)
        self._h_max = self._r.H0
        self._thrust_last = None
        return self._observation()

    def render(self, mode='human'):
        _, h, m = self._observation()

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            y = self._r.H_MAX_RENDER
            y0 = self._r.H0
            GY = (y-y0)/20
            Y = y-y0+GY
            H = Y/10
            W = H/10
            self.flame_offset = W/2

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(left=-Y/2, right=Y/2, bottom=y-Y, top=y)

            ground = rendering.make_polygon([(-Y/2,y0-GY), (-Y/2,y0), (Y/2,y0), (Y/2,y0-GY)])
            ground.set_color(.3, .6, .3)
            pad = rendering.make_polygon([(-3*W,y0-GY/3), (-3*W,y0), (3*W,y0), (3*W,y0-GY/3)])
            pad.set_color(.6, .6, .6)

            rocket = rendering.make_polygon([(-W/2,0), (-W/2,H), (W/2,H), (W/2,0)], filled=True)
            rocket.set_color(0, 0, 0)
            self.r_trans = rendering.Transform()
            rocket.add_attr(self.r_trans)

            self.make_fuel_poly = lambda mass: [
                (-W/2, 0),
                (-W/2, H*((mass-self._r.M1)/(self._r.M0-self._r.M1))),
                (W/2,  H*((mass-self._r.M1)/(self._r.M0-self._r.M1))),
                (W/2,0)
            ]
            self.fuel = rendering.make_polygon(self.make_fuel_poly(m), filled=True)
            self.fuel.set_color(.8, .1, .14)
            self.fuel.add_attr(self.r_trans)

            flame = rendering.make_circle(radius=W, res=30)
            flame.set_color(.96, 0.85, 0.35)
            self.f_trans = rendering.Transform()
            flame.add_attr(self.f_trans)

            flame_outer = rendering.make_circle(radius=2*W, res=30)
            flame_outer.set_color(.95, 0.5, 0.2)
            self.fo_trans = rendering.Transform()
            flame_outer.add_attr(self.fo_trans)

            for g in [ground, pad, rocket, self.fuel, flame_outer, flame]:
                self.viewer.add_geom(g)

        self.r_trans.set_translation(newx=0, newy=h)
        self.f_trans.set_translation(newx=0, newy=h)
        self.fo_trans.set_translation(newx=0, newy=h-self.flame_offset)

        self.fuel.v = self.make_fuel_poly(m)

        s = 0 if self._thrust_last is None else self._thrust_last/self._r.THRUST_MAX

        self.f_trans.set_scale(newx=s, newy=s)
        self.fo_trans.set_scale(newx=s, newy=s)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

_env = GoddardEnv()

def goddard_terminal(x, y):
    unnorm_x = _env.unnormalize(x[:3])
    v = unnorm_x[0]
    h = unnorm_x[1]
    m = unnorm_x[2]
    u = x[3]
    v2 = x[0]
    M1 = 0.6
    is_tank_empty = m <= M1
    is_descending = v2 < 0
    return is_tank_empty and is_descending


def goddard_reward(x, y):

    if not goddard_terminal(x, y):
        return 0
    unnorm_x = _env.unnormalize(x[:3])
    unnorm_y = _env.unnormalize(y)
    h0 = unnorm_x[1]
    h1 = unnorm_y[1]
    return max(h0, h1) - 1

class GoddardDefaultEnv(GoddardEnv):

    def __init__(self):
        super(GoddardDefaultEnv, self).__init__(rocket=Default())

class GoddardSaturnEnv(GoddardEnv):

    def __init__(self):
        super(GoddardSaturnEnv, self).__init__(rocket=SaturnV())

if __name__ == '__main__':
    env = GoddardDefaultEnv()
    obs = env.reset()
    unnorm = env.unnormalize(obs)
    norm = env.normalize(unnorm)
    assert np.allclose(norm, obs), f"norm={norm}, obs={obs}"
    done = False
    rewards = []
    while not done:
        next_obs, rew, done, info = env.step(env.action_space.sample())
        print(rew)
        rewards.append(rew)
    print(f"len(rewards)={len(rewards)}")

