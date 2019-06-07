from __future__ import print_function

# import gym
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# import other lib

import asyncore

from threading import Thread
import time 

import glob
import os
import sys


# Get time for get_reward
import time

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import ColorConverter as cc


import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref



try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, actor_filter):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

        self.velocity = 0
        self.distanceFromLane = 0
        self.angleFromLane = 0
        self.isCrossingLane = False
        self.lastCrossingFrame = 0

    def restartAtNewPos(self):
        spawn_points = self.map.get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.player.set_transform(spawn_point)
        self.player.set_velocity(carla.Vector3D())
        self.player.set_angular_velocity(carla.Vector3D())

        self.collision_sensor.sensor.destroy()
        self.lane_invasion_sensor.sensor.destroy()
        self.gnss_sensor.sensor.destroy()

        time.sleep(2.0)

        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self)
        self.gnss_sensor = GnssSensor(self.player)

        return
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager._index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager._transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)


    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager._index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager._transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)
        # calculate velocity
        if self.player is not None:
            v = self.player.get_velocity()
            self.velocity = math.sqrt(v.x**2 + v.y**2 + v.z**2) # in m/s

            # calculate distance from lane
            selfLoc = self.player.get_transform().location
            waypoint = self.map.get_waypoint(self.player.get_transform().location)
            if waypoint is not None:
                waypointLoc = waypoint.transform.location
                self.distanceFromLane = math.sqrt((selfLoc.x - waypointLoc.x)**2 + (selfLoc.y - waypointLoc.y)**2 + (selfLoc.z - waypointLoc.z)**2)

                next_waypoints = waypoint.next(1.0)
                if next_waypoints is not None and len(next_waypoints)>0:
                    nextLoc = next_waypoints[len(next_waypoints)-1].transform.location
                    waypointDiff = nextLoc - waypointLoc
                    
                    forward_vector = self.player.get_transform().get_forward_vector()

                    self.angleFromLane = math.degrees(math.atan2(waypointDiff.y,waypointDiff.x) - math.atan2(forward_vector.y,forward_vector.x))
                    # print(self.angleFromLane)
            if self.isCrossingLane == True and self.lastCrossingFrame<= 0:
                self.isCrossingLane = False
            else:
                self.lastCrossingFrame = self.lastCrossingFrame - 1
            
    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroySensors(self):
            self.camera_manager.sensor.destroy()
            self.camera_manager.sensor = None
            self.camera_manager._index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    currentIndex = world.camera_manager._index
                    world.destroySensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(currentIndex)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' % ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)

        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        if os.name == "posix":
            fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        else:
            fonts = [x for x in pygame.font.get_fonts() if 'arial' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        return
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        return
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item: # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        # lines = __doc__.split('\n')
        lines = ['','']
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self._hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


# class LaneInvasionSensor(object):
#     def __init__(self, parent_actor, hud):
#         self.sensor = None
#         self._parent = parent_actor
#         self._hud = hud
#         world = self._parent.get_world()
#         bp = world.get_blueprint_library().find('sensor.other.lane_detector')
#         self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
#         # We need to pass the lambda a weak reference to self to avoid circular
#         # reference.
#         weak_self = weakref.ref(self)
#         self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

#     @staticmethod
#     def _on_invasion(weak_self, event):
#         self = weak_self()
#         if not self:
#             return
#         text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
#         self._hud.notification('Crossed line %s' % ' and '.join(text))


# new lane invasion sensor for world
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, world):
        self.sensor = None
        self._parent = parent_actor
        self._world = world
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        self._world.isCrossingLane = True
        self._world.lastCrossingFrame = 20
        print("i m KG")

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self._index = None

        self.image_array = np.zeros((hud.dim[0],hud.dim[1],0))

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            self.image_array = array

            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self._recording:
            image.save_to_disk('_out/%08d' % image.frame_number)

class CarlaEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  ACTION_NAMES = ["steer", "throttle", "brake", "reverse"]
  STEER_LIMIT_LEFT = -1.0
  STEER_LIMIT_RIGHT = 1.0
  THROTTLE_MIN = 0.0
  THROTTLE_MAX = 1.0
  BRAKE_MIN = 0.0
  BRAKE_MAX = 1.0
  REVERSE_MIN = 0.0
  REVERSE_MAX = 1.0
  VAL_PER_PIXEL = 255

   # IP: 172.16.68.187 # SELF-HOST: 127.0.0.1 #172.16.68.140: LAPTOP HOST
  def __init__(self, host = '172.16.68.187', port = 2000, width = 640, height = 480, frame_to_store=3):
    print("init")

    self.host = host 
    self.port = port
    self.width = width
    self.height = height
    self.reward = 0

    self.action_space = spaces.Box(low=np.array([self.STEER_LIMIT_LEFT, self.THROTTLE_MIN, self.BRAKE_MIN, self.REVERSE_MIN]),
            high=np.array([self.STEER_LIMIT_RIGHT, self.THROTTLE_MAX, self.BRAKE_MAX, self.REVERSE_MAX]), dtype=np.float32 )

    self.threshold = 2
    high = np.array([
            self.threshold * 2,
            np.finfo(np.float32).max,
            self.threshold * 2,
            np.finfo(np.float32).max])
    self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, (height, width, 3), dtype=np.uint8)

    # simulation related variables.
    self.seed()

    # Frame to store
    self.frame_to_store = frame_to_store

    # Assign state
    self.state = []
    for i in range(0,frame_to_store):
      #self.state.append(np.zeros((self.width,self.height,0)))
      self.state.append(np.zeros((self.height,self.width,0)))

    print(self.state)
    self.world = None

    # carla world as viewer
    # self.game_loop()
    self.thread = Thread(target=self.game_loop)
    # self.thread.daemon = True
    self.thread.start()

    self.action = [0,0,0,0]

    self.velocity = 0
    self.distanceFromLane = 0
    self.previous_step_time = None


  def __del__(self):
    self.close()

  def close(self):
    self.state = []
    # print("close") 
    pass

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    # print(action)
    # self.state = (1,2,1,2)
    # self.action = action
    # print("step")
    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

    while True:
        try:
            reward = self.get_reward()
            done = self.is_done()
            self.action = action
            obs = self.get_observation()
        except Exception as e:
            print(e)
            continue
        break

    return obs, reward, done, {}
    
  def reset(self):
    print("reset")

    while True:
        try:
            obs = self.get_observation()

            if self.world is not None:
                self.world.restartAtNewPos()
            time.sleep(3)

        except Exception as e:
            print(e)
            continue

        break

    return obs

  def render(self, mode='human', close=False):
    # print("render")
    # self.camera_manager.image_array
    
    while True:
        try:
            obs = self.get_observation()

        except Exception as e:
            print(e)
            continue

        break

    return obs

    # return self.world.camera_manager.image_array

  def get_observation(self):
    if self.world is not None:
      # print("self.world is not None")
      return self.world.camera_manager.image_array
    else:
      # print("zero")
      # return np.zeros((self.width,self.height,0))
      return np.zeros((self.height,self.width,0))

  def get_observation_array(self):
    return self.state

  def update_observation(self):
    if(len(self.state) >0):
      self.state.pop(0)
      self.state.append(self.get_observation())

  def update_other_observation(self):
    if self.world is not None:
        
        self.velocity = self.world.velocity
        self.distanceFromLane = self.world.distanceFromLane
        self.angleFromLane = self.world.angleFromLane
        # sample for vehicles nearby
        # vehicles = self.world.world.get_actors().filter('vehicle.*')
        # if len(vehicles) > 1:
        #     t = self.world.player.get_transform()
        #     distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
        #     vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != self.world.player.id]
        #     for d, vehicle in sorted(vehicles):
        #         if d > 200.0:
        #             break
        #         vehicle_type = get_actor_display_name(vehicle, truncate=22)
        #         print('% 4dm %s' % (d, vehicle_type))



    # print(self.distanceFromLane)



  def cal_acceleration_mag(self):
    accelerate_3dVector = self.world.player.get_acceleration()
    my_forward_vector = self.world.player.get_transform().get_forward_vector()

    # acceleration_mag cal by DOT PRODUCT
    acceleration_mag = accelerate_3dVector.x * my_forward_vector.x + accelerate_3dVector.y * my_forward_vector.y + accelerate_3dVector.z * my_forward_vector.z

    return acceleration_mag

  def cal_velocity_mag(self):
      velocity_3dVector = self.world.player.get_velocity()

      velocity_mag = math.sqrt(velocity_3dVector.x**2 + velocity_3dVector.y**2 + velocity_3dVector.z**2)

      return velocity_mag

  def get_lane_deviation_penalty(self, time_passed):
      lane_deviation_penalty = 0
      lane_deviation_penalty = self.l(
              self.world.distanceFromLane, time_passed)
      return lane_deviation_penalty

  def l(self, lane_deviation, time_passed): #time_passed = update freq (step_time)
      lane_deviation_penalty = 0
      if lane_deviation < 0:
          raise ValueError('Lane deviation should be positive')
      if time_passed is not None and lane_deviation > 200:  # Tuned for Canyons spline - change for future maps
          lane_deviation_coeff = 0.1
          lane_deviation_penalty = lane_deviation_coeff * time_passed * lane_deviation ** 2 / 100.
      lane_deviation_penalty =min(max(lane_deviation_penalty, -1e2), 1e2)
      return lane_deviation_penalty




  def get_reward(self):
      reward = 0
      now = time.time()
      if self.previous_step_time != None:
          step_time = now - self.previous_step_time
      else:
          step_time=0

      if self.is_collide == True:
          return -100

      if self.world.isCrossingLane == True:
          return -1

      reward += -self.get_lane_deviation_penalty(step_time) + 0.1*self.cal_velocity_mag()

      self.previous_step_time = now

      return reward





  def get_reward_v0(self):
    # init variables
    v_brake = 10 # Speed that the car should start braking
    v_max = 20 # Speed Limit
    theta_thresohold = 5 # Thresohold from the center of the lane

    reward = 0


    # Rule 1: if v_t > v_brake AND action = brake: reward = reward + 1
    if self.cal_velocity_mag() > v_brake and self.action[2] >= 0.5:
        reward = reward + 1
    # Rule 2: if v_t > v_brake AND action != brake: reward = reward - 1
    if self.cal_velocity_mag() > v_brake and self.action[2] < 0.5:
        reward = reward - 1
    # Rule 3: if v_t < v_max AND action = accelerate: reward = reward + 1 (self.cal_acceleration_mag > 0 : Accelerate)
    if self.cal_velocity_mag() < v_max and self.cal_acceleration_mag() > 0:
        reward = reward + 1
    # Rule 3: if v_t < v_max AND action = accelerate: reward = reward - 1 (self.cal_acceleration_mag <= 0 : Decelerate/maintain same speed)
    if self.cal_velocity_mag() < v_max and self.cal_acceleration_mag() <= 0:
        reward = reward - 1
    # Rule 4: if Theta_t > theta_thresohold AND action = steer(positive value: Turn Right): reward = reward + 1 (Assume: Theta > Theta_Thresohold means drifting to the left)
    if self.world.angleFromLane > theta_thresohold and self.action[0] >= 0:
        reward = reward + 1
    if self.world.angleFromLane > theta_thresohold and self.action[0] < 0:
        reward = reward - 1
    if self.world.angleFromLane < -1*theta_thresohold and self.action[0] <= 0:
        reward = reward + 1
    if self.world.angleFromLane < -1*theta_thresohold and self.action[0] > 0:
        reward = reward - 1
    if self.is_collide == True:
        reward = - 100

    #self.world.angleFromLane
    #self.world.distanceFromLane

    #self.reward = reward

    return reward


  def get_reward_v1(self):
    # init variables
    v_brake = 10 # Speed that the car should start braking
    v_max = 20 # Speed Limit
    theta_thresohold = 5 # Thresohold from the center of the lane

    if self.is_collide == True:
        return -100

    if self.cal_velocity_mag() > v_brake and self.action[2] < 0.5:
        return -1

    if self.cal_velocity_mag() < v_max and self.cal_acceleration_mag() <= 0:
        return -1

    if self.world.angleFromLane > theta_thresohold and self.action[0] < 0:
        return -1

    if self.world.angleFromLane < -1*theta_thresohold and self.action[0] > 0:
        return - 1
    

    return abs(math.sin(math.radians(self.world.angleFromLane)))*self.cal_velocity_mag()
  

  def is_collide(self):
    if self.world is not None:
        if self.world.collision_sensor is not None:
            colhist = self.world.collision_sensor.get_collision_history()
            collision = [colhist[x + self.world.hud.frame_number - 200] for x in range(0, 200)]
            max_col = max(1.0, max(collision))
            if max_col > 1.0:
                return True
    return False


  def is_done(self):
    if self.world is not None:
        if self.world is not None:
            colhist = self.world.collision_sensor.get_collision_history()
            collision = [colhist[x + self.world.hud.frame_number - 200] for x in range(0, 200)]
            max_col = max(1.0, max(collision))
            if max_col > 1.0:
                return True
    return False

  def game_loop(self):

    # args
    args = type('', (), {})()
    args.host = self.host
    args.port = self.port
    args.width = self.width
    args.height = self.height
    args.filter = 'vehicle.audi.*'
    args.autopilot = False

    pygame.init()
    pygame.font.init()
    world = None
    

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        client.load_world('/Game/Carla/Maps/Town01')

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args.filter)
        # controller = KeyboardControl(world, args.autopilot)

        self.world = world
        self.hud = hud

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            # if controller.parse_events(client, world, clock):
                # return

            for event in pygame.event.get():
              if event.type == pygame.QUIT:
                return 

            #print(self.action)
            self._control = carla.VehicleControl()
            self._control.steer = float(self.action[0])
            self._control.throttle = float(self.action[1])
            self._control.brake = float(self.action[2]) > 0.5
            self._control.hand_brake = False
            self._control.reverse = float(self.action[3]) > 0.5
            world.player.apply_control(self._control)
            
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            self.update_observation()
            self.update_other_observation()

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()
