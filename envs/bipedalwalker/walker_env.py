# Copyright (c) OpenAI
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is an extended version of
# https://github.com/openai/gym/blob/master/gym/envs/box2d/bipedal_walker.py

import sys
import math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef,
                      polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding
from collections import namedtuple

EnvConfig = namedtuple('EnvConfig', [
    'name',
    'ground_roughness',
    'pit_gap',
    'stump_width',  'stump_height', 'stump_float',
    'stair_height', 'stair_width', 'stair_steps'
])

FPS = 50
SCALE = 30.0   # Affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [
    (-30, +9), (+6, +9), (+34, +1),
    (+34, -8), (-30, -8)
]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE)
                                 for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0)  # 0.99 bouncy

LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001)

STAIR_HEIGHT_EPS = 1e-2


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.hull == contact.fixtureA.body or self.env.hull == contact.fixtureB.body:
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True

    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False


class BipedalWalkerCustom(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __repr__(self):
        return "{}\nenv\n{}".format(self.__dict__, self.__dict__["np_random"].get_state())

    def __init__(self, env_config, seed=None):
        self.spec = None
        self.set_env_config(env_config)
        self.env_params = None
        self.env_seed = seed
        self._seed(seed)
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None

        self.prev_shaping = None
        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0),
                                         (1, 0),
                                         (1, -1),
                                         (0, -1)]),
            friction=FRICTION)

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0),
                                      (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        self._reset_env()

        high = np.array([np.inf] * 24)
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]), np.array([+1, +1, +1, +1]))
        self.observation_space = spaces.Box(-high, high)

    def re_init(self, env_config, seed):

        self.spec = None

        self.set_env_config(env_config)
        self._seed(seed)

        self.env_params = None

        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None

        self.prev_shaping = None
        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0),
                                         (1, 0),
                                         (1, -1),
                                         (0, -1)]),
            friction=FRICTION)

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0),
                                      (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        self._reset_env()

    def set_env_config(self, env_config):
        self.config = env_config

    def augment(self, params):
        self.env_params = params

    def _set_terrain_number(self):
        self.hardcore = False
        self.GRASS = 0
        self.STUMP, self.STAIRS, self.PIT = -1, -1, -1
        self._STATES_ = 1

        if self.config.stump_width and self.config.stump_height and self.config.stump_float:
            # STUMP exist
            self.STUMP = self._STATES_
            self._STATES_ += 1

        if self.config.stair_height and self.config.stair_width and self.config.stair_steps:
            # STAIRS exist
            self.STAIRS = self._STATES_
            self._STATES_ += 1

        if self.config.pit_gap:
            # PIT exist
            self.PIT = self._STATES_
            self._STATES_ += 1

        if self._STATES_ > 1:
            self.hardcore = True

    def save_env_def(self, filename):
        import json
        a = {'config': self.config._asdict(), 'seed': self.env_seed}
        with open(filename, 'w') as f:
            json.dump(a, f)

    def seed(self, seed=None):
        return self._seed(seed)

    def _seed(self, seed=None):
        self.env_seed = seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain:
            return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []
        self.world = None

    def _get_poly_stump(self, x, y, terrain_step):
        stump_width = self.np_random.randint(*self.config.stump_width)
        stump_height = self.np_random.uniform(*self.config.stump_height)
        stump_float = self.np_random.randint(*self.config.stump_float)
        # counter = np.ceil(stump_width)
        counter = stump_width
        countery = stump_height
        poly = [(x, y + stump_float * terrain_step),
                (x + stump_width * terrain_step, y + stump_float * terrain_step),
                (x + stump_width * terrain_step, y + countery * terrain_step + stump_float * terrain_step),
                (x, y + countery * terrain_step + stump_float * terrain_step), ]
        return poly

    def _generate_terrain(self, hardcore):
        #GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = self.GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        pit_diff = 0
        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == self.GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if self.env_params is not None and self.env_params.altitude_fn is not None:
                    y += velocity
                    if i > TERRAIN_STARTPAD:
                        mid = TERRAIN_LENGTH * TERRAIN_STEP / 2.
                        x_ = (x - mid) * np.pi / mid
                        y = TERRAIN_HEIGHT + self.env_params.altitude_fn((x_, ))[0]
                        if i == TERRAIN_STARTPAD+1:
                            y_norm = self.env_params.altitude_fn((x_, ))[0]
                        y -= y_norm
                else:
                    if i > TERRAIN_STARTPAD:
                        velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                    y += self.config.ground_roughness * velocity

            elif state == self.PIT and oneshot:
                pit_gap = 1.0 + self.np_random.uniform(*self.config.pit_gap)
                counter = np.ceil(pit_gap)
                pit_diff = counter - pit_gap

                poly = [
                    (x,              y),
                    (x + TERRAIN_STEP, y),
                    (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                    (x,              y - 4 * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [
                    (p[0] + TERRAIN_STEP * pit_gap, p[1]) for p in poly]
                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == self.PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * TERRAIN_STEP
                if counter == 1:
                    self.terrain_x[-1] = self.terrain_x[-1] - pit_diff * TERRAIN_STEP
                    pit_diff = 0

            elif state == self.STUMP and oneshot:
                # Sometimes this doesnt work due to randomness, 
                # so iterate until it does
                attempts = 0
                done = False
                while not done:
                    try:
                        poly = self._get_poly_stump(x, y, TERRAIN_STEP)
                        self.fd_polygon.shape.vertices = poly
                        done = True
                        self.env_seed -= int(attempts)
                    except:
                        self.seed(self.env_seed + int(1))
                        attempts += 1
                        if attempts > 10:
                            print("Stump issues: num attempts: ", attempts)
                            done = True

                t = self.world.CreateStaticBody(
                    fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

            elif state == self.STAIRS and oneshot:
                stair_height = self.np_random.uniform(
                    *self.config.stair_height)
                stair_slope = 1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(*self.config.stair_width)
                stair_steps = self.np_random.randint(*self.config.stair_steps)
                original_y = y

                if stair_height > STAIR_HEIGHT_EPS:
                    for s in range(stair_steps):
                        poly = [(x + (s * stair_width) * TERRAIN_STEP, y + (s * stair_height * stair_slope) * TERRAIN_STEP),
                                (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (s * stair_height * stair_slope) * TERRAIN_STEP),
                                (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (-stair_height + s * stair_height * stair_slope) * TERRAIN_STEP),
                                (x + (s * stair_width) * TERRAIN_STEP, y + (-stair_height + s * stair_height * stair_slope) * TERRAIN_STEP), ]

                        self.fd_polygon.shape.vertices = poly

                        t = self.world.CreateStaticBody(
                            fixtures=self.fd_polygon)
                        t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                        self.terrain.append(t)
                    counter = stair_steps * stair_width + 1

            elif state == self.STAIRS and not oneshot:
                s = stair_steps * stair_width - counter
                n = s // stair_width
                y = original_y + (n * stair_height * stair_slope) * TERRAIN_STEP - \
                    (stair_height if stair_slope == -1 else 0) * TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.randint(
                    TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if state == self.GRASS and hardcore:
                    state = self.np_random.randint(1, self._STATES_)
                    oneshot = True
                else:
                    state = self.GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1])
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_edge)
            color = (0.3, 1.0 if i % 2 == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly = []
        for i in range(TERRAIN_LENGTH // 20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH) * TERRAIN_STEP
            y = VIEWPORT_H / SCALE * 3 / 4
            poly = [
                (x + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5) + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                 y + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5) + self.np_random.uniform(0, 5 * TERRAIN_STEP))
                for a in range(5)]
            x1 = min([p[0] for p in poly])
            x2 = max([p[0] for p in poly])
            self.cloud_poly.append((poly, x1, x2))

    def _reset_env(self):
        self._destroy()
        self.world = Box2D.b2World()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        self._set_terrain_number()
        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=HULL_FD
        )
        self.hull.color1 = (0.5, 0.4, 0.9)
        self.hull.color2 = (0.3, 0.3, 0.5)
        self.hull.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        self.legs = []
        self.joints = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD
            )
            leg.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            leg.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=i,
                lowerAngle=-0.8,
                upperAngle=1.1,
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LOWER_FD
            )
            lower.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            lower.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction
        self.lidar = [LidarCallback() for _ in range(10)]

        return self._step(np.array([0, 0, 0, 0]))[0]

    def step(self, action):
        return self._step(action)

    def _step(self, action):
        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(
                SPEED_HIP * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(
                SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(
                SPEED_HIP * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(
                SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE)
            self.world.RayCast(
                self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            # Normal angles up to 0.5 here, but sure more is possible.
            self.hull.angle,
            2.0 * self.hull.angularVelocity / FPS,
            # Normalized to get -1..1 range
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].angle,
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0
        ]
        state += [l.fraction for l in self.lidar]
        assert len(state) == 24

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping = 130 * pos[0] / SCALE
        # keep head straight, other than that and falling, any behavior is unpunished
        shaping -= 5.0 * abs(state[0])

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less
        done = False
        finish = False
        if self.game_over or pos[0] < 0:
            reward = -100
            done = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            done = True
            finish = True
        return np.array(state), reward, done, {"finish": finish}

    def render(self, *args, **kwargs):
        return self._render(*args, **kwargs)

    def _render(self, mode='level', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W /
                               SCALE + self.scroll, 0, VIEWPORT_H / SCALE)

        self.viewer.draw_polygon([
            (self.scroll,                  0),
            (self.scroll + VIEWPORT_W / SCALE, 0),
            (self.scroll + VIEWPORT_W / SCALE, VIEWPORT_H / SCALE),
            (self.scroll,                  VIEWPORT_H / SCALE),
        ], color=(0.9, 0.9, 1.0))
        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2:
                continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE:
                continue
            self.viewer.draw_polygon(
                [(p[0] + self.scroll / 2, p[1]) for p in poly], color=(1, 1, 1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE:
                continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            l = self.lidar[i] if i < len(
                self.lidar) else self.lidar[len(self.lidar) - i - 1]
            self.viewer.draw_polyline(
                [l.p1, l.p2], color=(1, 0, 0), linewidth=1)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(
                        f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(
                        f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(
                        path, color=obj.color2, linewidth=2)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50 / SCALE
        x = TERRAIN_STEP * 3
        self.viewer.draw_polyline(
            [(x, flagy1), (x, flagy2)], color=(0, 0, 0), linewidth=2)
        f = [(x, flagy2), (x, flagy2 - 10 / SCALE),
             (x + 25 / SCALE, flagy2 - 5 / SCALE)]
        self.viewer.draw_polygon(f, color=(0.9, 0.2, 0))
        self.viewer.draw_polyline(f + [f[0]], color=(0, 0, 0), linewidth=2)

        return_rgb_array = mode in ['rgb_array', 'level']

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
