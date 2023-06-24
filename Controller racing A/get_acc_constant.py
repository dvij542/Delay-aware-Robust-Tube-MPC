# -*- coding: utf-8 -*-

#!/usr/bin/env python3

'''
This code implements the high level Delay aware MPC controller 
for following higher level trajectory from motion planner
'''
from __future__ import print_function
from __future__ import division
# Imports
if True :
    from adaptive_kalman_filter import update_delay_time_estimate
    from casadi import *
    from IRIS import *
    from inv_set_calc import *
    from adaptive_kalman_filter import *
    import matplotlib.pyplot as plt
    
    import live_plotter as lv
    import numpy as np
    import math
    # import rospy
    # import math
    from FrenetOptimalTrajectory.frenet_optimal_trajectory import *
    # import rticonnextdds_connector as rti
    from sys import path as sys_path
    from os import path as os_path, times
    from beginner_tutorials.msg import custom_msg
    # import util_funcs as utils
    import rospy

# manualcontrol.py stuff
if True :


    # ==============================================================================
    # -- find carla module ---------------------------------------------------------
    # ==============================================================================


    import glob
    import os
    import sys

    try:
        sys.path.append(glob.glob('../../../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass


    # ==============================================================================
    # -- imports -------------------------------------------------------------------
    # ==============================================================================


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
        # from math import *
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
        from pygame.locals import K_g
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
        return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


    # ==============================================================================
    # -- World ---------------------------------------------------------------------
    # ==============================================================================


    class World(object):
        def __init__(self, carla_world, hud, args):
            self.world = carla_world
            self.actor_role_name = args.rolename
            try:
                self.map = self.world.get_map()
            except RuntimeError as error:
                print('RuntimeError: {}'.format(error))
                print('  The server could not send the OpenDRIVE (.xodr) file:')
                print('  Make sure it exists, has the same name of your town, and is correct.')
                sys.exit(1)
            self.hud = hud
            self.player = None
            self.collision_sensor = None
            self.lane_invasion_sensor = None
            self.gnss_sensor = None
            self.imu_sensor = None
            self.radar_sensor = None
            self.camera_manager = None
            self._weather_presets = find_weather_presets()
            self._weather_index = 0
            self._actor_filter = args.filter
            self._gamma = args.gamma
            self.restart()
            self.world.on_tick(hud.on_world_tick)
            self.recording_enabled = False
            self.recording_start = 0

        def restart(self):
            self.player_max_speed = 1.589
            self.player_max_speed_fast = 3.713
            # Keep same camera config if the camera manager exists.
            cam_index = self.camera_manager.index if self.camera_manager is not None else 0
            cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
            # Get a random blueprint.
            blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
            blueprint.set_attribute('role_name', self.actor_role_name)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible', 'true')
            # set the max speed
            if blueprint.has_attribute('speed'):
                self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
                self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
            else:
                print("No recommended values for 'speed' attribute")
            # Spawn the player.
            if self.player is not None:
                spawn_point = self.player.get_transform()
                spawn_point.location.z += 2.0
                spawn_point.rotation.roll = 0.0
                spawn_point.rotation.pitch = 0.0
                self.destroy()
                self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            while self.player is None:
                if not self.map.get_spawn_points():
                    print('There are no spawn points available in your map/town.')
                    print('Please add some Vehicle Spawn Point to your UE4 scene.')
                    sys.exit(1)
                spawn_points = self.map.get_spawn_points()
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
                print("Spawn point : ", spawn_point)
                spawn_point.location.x = -364.7
                spawn_point.location.y = 65.3
                spawn_point.rotation.yaw = 90

                self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            # Set up the sensors.
            
            self.collision_sensor = CollisionSensor(self.player, self.hud)
            self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
            self.gnss_sensor = GnssSensor(self.player)
            self.imu_sensor = IMUSensor(self.player)
            self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
            self.camera_manager.transform_index = cam_pos_index
            self.camera_manager.set_sensor(cam_index, notify=False)
            actor_type = get_actor_display_name(self.player)
            self.hud.notification(actor_type)
            actors = self.world.get_actors()

            vehicle = self.player#random.choice([actor for actor in actors if 'vehicle' in actor.type_id])
            front_left_wheel  = carla.WheelPhysicsControl(tire_friction=2.2, damping_rate=1.5, max_steer_angle=70.0)#, long_stiff_value=1000)
            front_right_wheel = carla.WheelPhysicsControl(tire_friction=2.2, damping_rate=1.5, max_steer_angle=70.0)#, long_stiff_value=1000)
            rear_left_wheel   = carla.WheelPhysicsControl(tire_friction=2.2, damping_rate=1.5, max_steer_angle=0.0)#,  long_stiff_value=1000)
            rear_right_wheel  = carla.WheelPhysicsControl(tire_friction=2.2, damping_rate=1.5, max_steer_angle=0.0)#,  long_stiff_value=1000)
            wheels = [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]
            physics_control = vehicle.get_physics_control()
            physics_control.torque_curve = [carla.Vector2D(x=0.000000, y=600.000000), carla.Vector2D(x=1890.760742, y=600.000000), carla.Vector2D(x=5729.577637, y=600.000000)]
            physics_control.drag_coefficient = 0.3
            physics_control.wheels = wheels
            vehicle.apply_physics_control(physics_control)
            print(physics_control)
            

        def next_weather(self, reverse=False):
            self._weather_index += -1 if reverse else 1
            self._weather_index %= len(self._weather_presets)
            preset = self._weather_presets[self._weather_index]
            self.hud.notification('Weather: %s' % preset[1])
            self.player.get_world().set_weather(preset[0])

        def toggle_radar(self):
            if self.radar_sensor is None:
                self.radar_sensor = RadarSensor(self.player)
            elif self.radar_sensor.sensor is not None:
                self.radar_sensor.sensor.destroy()
                self.radar_sensor = None

        def tick(self, clock):
            self.hud.tick(self, clock)

        def render(self, display):
            self.camera_manager.render(display)
            self.hud.render(display)

        def destroy_sensors(self):
            self.camera_manager.sensor.destroy()
            self.camera_manager.sensor = None
            self.camera_manager.index = None

        def destroy(self):
            if self.radar_sensor is not None:
                self.toggle_radar()
            actors = [
                self.camera_manager.sensor,
                self.collision_sensor.sensor,
                self.lane_invasion_sensor.sensor,
                self.gnss_sensor.sensor,
                self.imu_sensor.sensor,
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
                    elif event.key == K_g:
                        world.toggle_radar()
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
                        currentIndex = world.camera_manager.index
                        world.destroy_sensors()
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
                            world.hud.notification('%s Transmission' %
                                                ('Manual' if self._control.manual_gear_shift else 'Automatic'))
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
                    self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
                world.player.apply_control(self._control)

        def _parse_vehicle_keys(self, keys, milliseconds):
            self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
            steer_increment = 5e-4 * milliseconds
            if keys[K_LEFT] or keys[K_a]:
                if self._steer_cache > 0:
                    self._steer_cache = 0
                else:
                    self._steer_cache -= steer_increment
            elif keys[K_RIGHT] or keys[K_d]:
                if self._steer_cache < 0:
                    self._steer_cache = 0
                else:
                    self._steer_cache += steer_increment
            else:
                self._steer_cache = 0.0
            self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
            self._control.steer = round(self._steer_cache, 1)
            self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
            self._control.hand_brake = keys[K_SPACE]

        def _parse_walker_keys(self, keys, milliseconds, world):
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
                self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
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
            font_name = 'courier' if os.name == 'nt' else 'mono'
            fonts = [x for x in pygame.font.get_fonts() if font_name in x]
            default_font = 'ubuntumono'
            mono = default_font if default_font in fonts else fonts[0]
            mono = pygame.font.match_font(mono)
            self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
            self._notifications = FadingText(font, (width, 40), (0, height - 40))
            self.help = HelpText(pygame.font.Font(mono, 24), width, height)
            self.server_fps = 0
            self.frame = 0
            self.simulation_time = 0
            self._show_info = True
            self._info_text = []
            self._server_clock = pygame.time.Clock()

        def on_world_tick(self, timestamp):
            self._server_clock.tick()
            self.server_fps = self._server_clock.get_fps()
            self.frame = timestamp.frame
            self.simulation_time = timestamp.elapsed_seconds

        def tick(self, world, clock):
            self._notifications.tick(world, clock)
            if not self._show_info:
                return
            t = world.player.get_transform()
            v = world.player.get_velocity()
            c = world.player.get_control()
            compass = world.imu_sensor.compass
            heading = 'N' if compass > 270.5 or compass < 89.5 else ''
            heading += 'S' if 90.5 < compass < 269.5 else ''
            heading += 'E' if 0.5 < compass < 179.5 else ''
            heading += 'W' if 180.5 < compass < 359.5 else ''
            colhist = world.collision_sensor.get_collision_history()
            collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
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
                u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
                'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
                'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
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
                    if item:  # At this point has to be a str.
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
            lines = __doc__.split('\n')
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
            self.history = []
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.collision')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

        def get_collision_history(self):
            history = collections.defaultdict(int)
            for frame, intensity in self.history:
                history[frame] += intensity
            return history

        @staticmethod
        def _on_collision(weak_self, event):
            self = weak_self()
            if not self:
                return
            actor_type = get_actor_display_name(event.other_actor)
            self.hud.notification('Collision with %r' % actor_type)
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.history.append((event.frame, intensity))
            if len(self.history) > 4000:
                self.history.pop(0)


    # ==============================================================================
    # -- LaneInvasionSensor --------------------------------------------------------
    # ==============================================================================


    class LaneInvasionSensor(object):
        def __init__(self, parent_actor, hud):
            self.sensor = None
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
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
            lane_types = set(x.type for x in event.crossed_lane_markings)
            text = ['%r' % str(x).split()[-1] for x in lane_types]
            self.hud.notification('Crossed line %s' % ' and '.join(text))


    # ==============================================================================
    # -- GnssSensor ----------------------------------------------------------------
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
    # -- IMUSensor -----------------------------------------------------------------
    # ==============================================================================


    class IMUSensor(object):
        def __init__(self, parent_actor):
            self.sensor = None
            self._parent = parent_actor
            self.accelerometer = ()
            self.gyroscope = ()
            self.compass = 0.0
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.imu')
            self.sensor = world.spawn_actor(
                bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

        @staticmethod
        def _IMU_callback(weak_self, sensor_data):
            self = weak_self()
            if not self:
                return
            limits = (-99.9, 99.9)
            self.accelerometer = (
                max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
                max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
                max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
            self.gyroscope = (
                max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
                max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
                max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
            self.compass = math.degrees(sensor_data.compass)


    # ==============================================================================
    # -- RadarSensor ---------------------------------------------------------------
    # ==============================================================================


    class RadarSensor(object):
        def __init__(self, parent_actor):
            self.sensor = None
            self._parent = parent_actor
            self.velocity_range = 7.5 # m/s
            world = self._parent.get_world()
            self.debug = world.debug
            bp = world.get_blueprint_library().find('sensor.other.radar')
            bp.set_attribute('horizontal_fov', str(35))
            bp.set_attribute('vertical_fov', str(20))
            self.sensor = world.spawn_actor(
                bp,
                carla.Transform(
                    carla.Location(x=2.8, z=1.0),
                    carla.Rotation(pitch=5)),
                attach_to=self._parent)
            # We need a weak reference to self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

        @staticmethod
        def _Radar_callback(weak_self, radar_data):
            self = weak_self()
            if not self:
                return
            # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
            # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
            # points = np.reshape(points, (len(radar_data), 4))

            current_rot = radar_data.transform.rotation
            for detect in radar_data:
                azi = math.degrees(detect.azimuth)
                alt = math.degrees(detect.altitude)
                # The 0.25 adjusts a bit the distance so the dots can
                # be properly seen
                fw_vec = carla.Vector3D(x=detect.depth - 0.25)
                carla.Transform(
                    carla.Location(),
                    carla.Rotation(
                        pitch=current_rot.pitch + alt,
                        yaw=current_rot.yaw + azi,
                        roll=current_rot.roll)).transform(fw_vec)

                def clamp(min_v, max_v, value):
                    return max(min_v, min(value, max_v))

                norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
                r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
                g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
                b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
                self.debug.draw_point(
                    radar_data.transform.location + fw_vec,
                    size=0.075,
                    life_time=0.06,
                    persistent_lines=False,
                    color=carla.Color(r, g, b))

    # ==============================================================================
    # -- CameraManager -------------------------------------------------------------
    # ==============================================================================


    class CameraManager(object):
        def __init__(self, parent_actor, hud, gamma_correction):
            self.sensor = None
            self.surface = None
            self._parent = parent_actor
            self.hud = hud
            self.recording = False
            bound_y = 0.5 + self._parent.bounding_box.extent.y
            Attachment = carla.AttachmentType
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
            self.transform_index = 1
            self.sensors = [
                ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
                ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
                ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
                ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
                ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
                ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                    'Camera Semantic Segmentation (CityScapes Palette)', {}],
                ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}],
                ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                    {'lens_circle_multiplier': '3.0',
                    'lens_circle_falloff': '3.0',
                    'chromatic_aberration_intensity': '0.5',
                    'chromatic_aberration_offset': '0'}]]
            world = self._parent.get_world()
            bp_library = world.get_blueprint_library()
            for item in self.sensors:
                bp = bp_library.find(item[0])
                if item[0].startswith('sensor.camera'):
                    bp.set_attribute('image_size_x', str(hud.dim[0]))
                    bp.set_attribute('image_size_y', str(hud.dim[1]))
                    if bp.has_attribute('gamma'):
                        bp.set_attribute('gamma', str(gamma_correction))
                    for attr_name, attr_value in item[3].items():
                        bp.set_attribute(attr_name, attr_value)
                elif item[0].startswith('sensor.lidar'):
                    bp.set_attribute('range', '50')
                item.append(bp)
            self.index = None

        def toggle_camera(self):
            self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
            self.set_sensor(self.index, notify=False, force_respawn=True)

        def set_sensor(self, index, notify=True, force_respawn=False):
            index = index % len(self.sensors)
            needs_respawn = True if self.index is None else \
                (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
            if needs_respawn:
                if self.sensor is not None:
                    self.sensor.destroy()
                    self.surface = None
                self.sensor = self._parent.get_world().spawn_actor(
                    self.sensors[index][-1],
                    self._camera_transforms[self.transform_index][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[self.transform_index][1])
                # We need to pass the lambda a weak reference to self to avoid
                # circular reference.
                weak_self = weakref.ref(self)
                self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
            if notify:
                self.hud.notification(self.sensors[index][2])
            self.index = index

        def next_sensor(self):
            self.set_sensor(self.index + 1)

        def toggle_recording(self):
            self.recording = not self.recording
            self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

        def render(self, display):
            if self.surface is not None:
                display.blit(self.surface, (0, 0))

        @staticmethod
        def _parse_image(weak_self, image):
            self = weak_self()
            if not self:
                return
            if self.sensors[self.index][0].startswith('sensor.lidar'):
                points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 3), 3))
                lidar_data = np.array(points[:, :2])
                lidar_data *= min(self.hud.dim) / 100.0
                lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
                lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
                lidar_data = lidar_data.astype(np.int32)
                lidar_data = np.reshape(lidar_data, (-1, 2))
                lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
                lidar_img = np.zeros((lidar_img_size), dtype = int)
                lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
                self.surface = pygame.surfarray.make_surface(lidar_img)
            else:
                image.convert(self.sensors[self.index][1])
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            if self.recording:
                image.save_to_disk('_out/%08d' % image.frame)


    
    
    # ==============================================================================
    # -- game_loop() ---------------------------------------------------------------
    # ==============================================================================

# Parameters
if True :
    T = .1     # Planning time horizon
    N = 8 # No of control intervals
    speed = 20 # Target speed of the vehicle
    save_path_after = 300 # Start saving path after this much time from start (in s)
    wait_time = -1 # If -1, adaptive kalman filter will be used to predict it
    extra_time_to_wait = 0.0 # Wait for this additional extra time (in s) after each computation cycle to simulate and verify safety on slow systems
    Td = 0.04 # Actuator processing delay (in s)
    steer_to_wheel_ratio = (70*math.pi/180)
    has_start=False
    curr_time_est = 0  # Estimated mean value of computation time
    curr_time_est_var = 0 # Estimated variance of computation time

    g_const = 9.8
    mu = 1 # Friction constant (F_{y,max} = mu*mass*g_const)
    scenario = 'static'

    obstacle_points = np.array([[22,-18],[22,-10],[32,-10],[32,-18]]) # Static obstacle

    L = 3 # Length of the vehicle in m
    Ka = 4.25 # Pedal constant (F_x = Ka*mass*pedal_value)
    Kf = -0.25 # Friction resistance (F_{friction,x} = Kf*mass)
    vehicle_footprint = [[3,1],[-1,1],[-1,-1],[3,-1]] # Vehicle dimensions in m

    Q_robust = np.matrix(np.diag(np.array([1,1,1,1]))) 
    R_robust = np.matrix(np.diag(np.array([.1,.1])))

    ############ Calculated offline from inv_set_calc.py (Last commented part) ###############
    steering_limits = [0.5511084632063113, 0.5511084632063113, 0.5511084632063113, \
        0.5185237587941808, 0.5511084632063113, 0.5727896385850489, 0.5896766286658156, \
        0.6037252485785009, 0.616120511291855, 0.6266117297066048, 0.6266117297066048]
    acc_limits = [3.7014163903050914, 3.700715491966788, 3.7157664426919617, \
        3.7346625840889347, 3.751783194067104, 3.7654178037240746, 3.7756355027001733, \
        3.7829216295990125, 3.7880532616963265, 3.791426044016998, 3.791426044016998]
    #################### Function of speeds (step size 1 m/s) ################################

    DONT_CONSIDER_COMP_DELAY = False # If True, computation delay compensation will not be considered
    DONT_CONSIDER_STEERING_DYNAMICS = False # If True, actuator dynamic delay compensation will not be considered
    file_centre_line = "./racetrack_waypoints.txt"  # File to read the global reference line, if None then centre line will be taken
    file_path_follow = "./waypoints_new.csv"  # File to read the global reference line, if None then centre line will be taken
    file_new_path = "./coordinates_nc2.txt" # File in which the new coordinates will be saved
    Q_along=2  # Weight for progress along the road
    Q_dist=0  # Weight for distance for the center of the road
    penalty_out_of_road = 6 # Penalise for planning path outside the road
    no_iters = 3
    max_no_of_vehicles = 4
    max_vehicle_id = 10

# Global variables for internal communication (Don't change)
if True :
    buff_con = [0]*N # buffer sequence of commands
    gt_steering = 0 # Variable to communicate current value of steering angle
    inv_set = [] # Invariant set, Z
    time_estimates = []
    planned_paths = []
    time_to_finish = 0
    FIRST_TIME = True

# Definitions of optimization objective 
if True : 
    ###########   States    ####################

    x=SX.sym('x')
    y=SX.sym('y')
    theta=SX.sym('theta')
    pedal=SX.sym('pedal')
    delta_ac=SX.sym('delta_ac')
    v=SX.sym('v')
    states=vertcat(x,y,theta,v,delta_ac)
    delta=SX.sym('delta')
    controls=vertcat(pedal,delta)
    EPSILON = 0
    if DONT_CONSIDER_STEERING_DYNAMICS :
        K = 1/T
    else :
        K = (1-math.e**(-30*T))/T

    ###########    Model   #####################
    rhs=[
            v*cos(theta+EPSILON),
            v*sin(theta+EPSILON),
            v*tan(delta*steer_to_wheel_ratio)/L,
            Ka*pedal+Kf,
            K*(delta-delta_ac)
        ]                                            
    rhs=vertcat(*rhs)
    f=Function('f',[states,controls],[rhs])

    n_states=5
    n_controls=2
    U=SX.sym('U',n_controls,N)
    g=SX.sym('g',(1+len(vehicle_footprint))*(N+1))
    P=SX.sym('P',n_states + 2*N + 3)
    X=SX.sym('X',n_states,(N+1))
    X[:,0]=P[0:n_states]         


    for k in range(0,N,1):
        st=X[:,k]
        con=U[:,k]
        f_value=f(st,con)
        st_next=st+(T*f_value)
        X[:,k+1]=st_next

    ############### Objective function ################

    ff=Function('ff',[U,P],[X])

    obj=0
    Q=SX([[1,0],
        [0,1]])
    Q_speed = 10

    R=SX([[10,0],
        [0,10]])

    R2=SX([[100,0],
        [0,100]])

    for k in range(0,N,1):
        st=X[:,k+1]
        con=U[:,k]
        j = n_states + 2*k
        obj=obj+(((st[:2]- P[j:(j+2)]).T)@Q)@(st[:2]- P[j:(j+2)]) + Q_speed*(st[3]-speed)**2 + con.T@R@con

    for k in range(0,N-1,1):
        prev_con=U[:,k]
        next_con=U[:,k+1]
        obj=obj+(prev_con- next_con).T@R2@(prev_con- next_con)

    OPT_variables = reshape(U,2*N,1)

    a_eq = P[-3]
    b_eq = P[-2]
    c_eq = P[-1]

    for v in range(0,len(vehicle_footprint)) :
        for k in range (0,N+1,1): 
            g[k+(N+1)*v] = 5 # a_eq*(X[0,k]+vehicle_footprint[v][0]*cos(X[2,k])\
                # -vehicle_footprint[v][1]*sin(X[2,k])) + b_eq*(X[1,k]+\
                # vehicle_footprint[v][0]*sin(X[2,k]) + \
                # vehicle_footprint[v][1]*cos(X[2,k])) + c_eq   

    for k in range (0,N+1,1): 
        g[k+(N+1)*len(vehicle_footprint)] = 0#-X[3,k]**2*tan(X[2,k])/L

    nlp_prob = {'f': obj, 'x':OPT_variables, 'p': P,'g':g}
    options = {
                'ipopt.print_level' : 0,
                'ipopt.max_iter' : 150,
                'ipopt.mu_init' : 0.01,
                'ipopt.tol' : 1e-8,
                'ipopt.warm_start_init_point' : 'yes',
                'ipopt.warm_start_bound_push' : 1e-9,
                'ipopt.warm_start_bound_frac' : 1e-9,
                'ipopt.warm_start_slack_bound_frac' : 1e-9,
                'ipopt.warm_start_slack_bound_push' : 1e-9,
                'ipopt.mu_strategy' : 'adaptive',
                'print_time' : False,
                'verbose' : False,
                'expand' : True
            }

    solver=nlpsol("solver","ipopt",nlp_prob,options)

    u0=np.random.rand(N,2)
    x0=reshape(u0,2*N,1)

# Utility functions
if True :
    def dist(a, x, y):
        return (((a.pose.position.x - x)**2) + ((a.pose.position.y - y)**2))**0.5

    def path_length_distance(a,b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

    def calc_path_length(x_p):
        # global path_length
        path_length = []
        for i in range(len(x_p)):
            if i == 0:
                path_length.append(0)
            else:
                path_length.append(path_length[i-1] + path_length_distance(x_p[i], x_p[i-1]))
        return path_length

    def convert_xyzw_to_rpy(x, y, z, w):
            """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)
            """
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll_x = math.atan2(t0, t1)
        
            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = math.asin(t2)
        
            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw_z = math.atan2(t3, t4)
        
            return roll_x, pitch_y, yaw_z # in radians

    # Preprocessing to get the trackable path by the vehicle (for MPC) at current speed, for N steps at T step length
    def get_path(x_bot, y_bot, theta_bot, x_p, N,speed,T) :
        out_path = []
        path_lengths = calc_path_length(x_p)
        distances = []   
        # print(len(x_p)) 
        for i in range(len(x_p)):
            a = x_p[i]
            distances += [path_length_distance([x_bot,y_bot],a)]
        # print(distances)
        
        ep = min(distances)
        total_index=len(x_p)
        cp = distances.index(ep)
        curr_dist = path_lengths[cp]
        s0 = curr_dist
        i = cp
        
        # In case of very sparsely present points, divide segments into multiple parts to get closest point
        new_dists = []
        if cp > 0 :
            available_length_l = path_lengths[cp] - path_lengths[cp-1]
        else :
            available_length_l = 0
        
        if cp < len(path_lengths) - 1 :
            available_length_r = path_lengths[cp+1] - path_lengths[cp]
        else :
            available_length_r = 0
        
        no_of_segs_l = int(available_length_l/(speed*T)) 
        no_of_segs_r = int(available_length_r/(speed*T)) 
        seg_len_l = available_length_l/max(no_of_segs_l,1)
        seg_len_r = available_length_r/max(no_of_segs_r,1)
        for s in range(no_of_segs_l) :
            x1,y1 = x_p[cp-1][0], x_p[cp-1][1]
            x2,y2 = x_p[cp][0], x_p[cp][1]
            xs,ys = x1 + (x2-x1)*(seg_len_l/available_length_l)*(s+1), y1 + (y2-y1)*(seg_len_l/available_length_l)*(s+1)
            new_dists += [((xs-x_bot)**2 + (ys-y_bot)**2)**0.5]
        new_dists.append(ep)
        for s in range(no_of_segs_r) :
            x1,y1 = x_p[cp][0], x_p[cp][1]
            x2,y2 = x_p[cp+1][0], x_p[cp+1][1]
            xs,ys = x1 + (x2-x1)*(seg_len_r/available_length_r)*(s+1), y1 + (y2-y1)*(seg_len_r/available_length_r)*(s+1)
            new_dists += [((xs-x_bot)**2 + (ys-y_bot)**2)**0.5]
        min_ni = new_dists.index(min(new_dists))
        if min_ni < no_of_segs_l :
            s = min_ni
            x1,y1 = x_p[cp-1][0], x_p[cp-1][1]
            x2,y2 = x_p[cp][0], x_p[cp][1]
            xs,ys = x1 + (x2-x1)*(seg_len_l/available_length_l)*(s+1), y1 + (y2-y1)*(seg_len_l/available_length_l)*(s+1)
            
            v1 = x_p[cp-1][2]
            v2 = x_p[cp][2]
            vs = v1 + (v2-v1)*(seg_len_l/available_length_l)*(s+1)
            
            # pose_temp=PoseStamped()
            # pose_temp.pose.position.x = xs
            # pose_temp.pose.position.y = ys
            x_p.insert(cp,[xs,ys,vs])
            s0 = path_lengths[cp-1] + seg_len_l*(s+1)
            path_lengths.insert(cp,path_lengths[cp-1] + seg_len_l*(s+1))
        if min_ni > no_of_segs_l :
            s = min_ni - no_of_segs_l - 1
            x1,y1 = x_p[cp][0], x_p[cp][1]
            x2,y2 = x_p[cp+1][0], x_p[cp+1][1]
            xs,ys = x1 + (x2-x1)*(seg_len_r/available_length_r)*(s+1), y1 + (y2-y1)*(seg_len_r/available_length_r)*(s+1)
            
            v1 = x_p[cp-1][2]
            v2 = x_p[cp][2]
            vs = v1 + (v2-v1)*(seg_len_r/available_length_r)*(s+1)
            
            x_p.insert(cp+1,[xs,ys,vs])
            s0 = path_lengths[cp] + seg_len_r*(s+1)
            path_lengths.insert(cp+1,path_lengths[cp] + seg_len_r*(s+1))
            cp = cp + 1
        i = cp
        
        # Building the path
        for j in range(N+1) :
            req_dist = (j)*speed*T
            k = i
            # print(k,req_dist)
            while(k<len(path_lengths) and path_lengths[k]-path_lengths[cp]<req_dist ) :
                k += 1
            if k>=len(path_lengths) :
                k = len(path_lengths) - 1
                out_path.append(np.array([x_p[k][0],x_p[k][1]]))
                continue
            # print(k)
            a = req_dist + path_lengths[cp] - path_lengths[k-1]
            b = path_lengths[k] - req_dist - path_lengths[cp]
            X1 = np.array([x_p[k-1][0],x_p[k-1][1],x_p[k-1][2]])
            X2 = np.array([x_p[k][0],x_p[k][1],x_p[k][2]])
            X = X1*b/(a+b) + X2*a/(a+b)
            out_path.append(X)
            i = k-1
        # print("Path : ", out_path)
        X1 = out_path[0]
        X2 = out_path[-1]
        x1_,y1_ = X1[0]-x_bot,X1[1]-y_bot
        x2_,y2_ = X2[0]-x_bot,X2[1]-y_bot
        x1 = x1_*cos(theta_bot) + y1_*sin(theta_bot)
        y1 = y1_*cos(theta_bot) - x1_*sin(theta_bot)
        x2 = x2_*cos(theta_bot) + y2_*sin(theta_bot)
        y2 = y2_*cos(theta_bot) - x2_*sin(theta_bot)
        dist = ((y1-y2)**2 + (x2-x1)**2)**(1/2)
        a = (y1-y2)/dist
        b = (x2-x1)/dist
        c = (y2*x1 - y1*x2)/dist
        return np.array(out_path[1:]),[a,b,c,s0,X1[2]]

    # Get shifted initial pose of the ego vehicle
    def get_future_state_est(current_pose,buff_con,curr_time_est) :
        # print("Pose before : ", np.array(current_pose))
        if DONT_CONSIDER_COMP_DELAY :
            return np.array(current_pose)
        itr = 0
        while (curr_time_est > T) :
            # print("Start ", buff_con)
            if buff_con[0] !=0 :
                # print("Was ", buff_con[itr,:])
                f_value=f(current_pose,buff_con[itr,:])
            else :
                f_value=f(current_pose,float(buff_con[itr]))
            current_pose = list(np.array(current_pose)+np.array(T*f_value)[:,0])
            itr = itr + 1
            curr_time_est = curr_time_est - T
        # print("Was ", buff_con[itr])
        # print("Used ", float(buff_con[itr]))
        if buff_con[0] !=0 :
            # print("Was ", buff_con[itr,:])
            f_value=f(current_pose,buff_con[itr,:])
        else :
            f_value=f(current_pose,float(buff_con[itr]))# print("f_value :", f_value)
        # print(curr_time_est, rospy.get_time())
        # print(np.array(curr_time_est*f_value)[:,0])
        return np.array(current_pose)+np.array(curr_time_est*f_value)[:,0]

file_path = os_path.dirname(os_path.realpath(__file__))
sys_path.append(file_path + "/../../../")

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    # global inv_set
    
    # inv_set = get_inv_set(T,0,2,Q_robust,R_robust,N,without_steering=True)
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        # world_load = client.load_world(args.map)
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)
        settings = client.get_world().get_settings()
        settings.synchronous_mode = False # Enables synchronous mode
        settings.fixed_delta_seconds=0.03
        client.get_world().apply_settings(settings)
        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                break
            world.tick(clock)
            world.render(display)
            client.get_world().tick()
            pygame.display.flip()
        print("Starting automated waypoint follower")
        lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
        FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
        FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
        PLOT_LEFT          = 0.1    # in fractions of figure width and height
        PLOT_BOT           = 0.1    
        PLOT_WIDTH         = 0.8
        PLOT_HEIGHT        = 0.8
        TOTAL_EPISODE_FRAMES = 1000
        trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
                title='Vehicle Trajectory',
                figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
                edgecolor="black",
                rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

        waypoints = np.loadtxt('racetrack_waypoints.txt', delimiter=',')
        waypoints[:,1] = -waypoints[:,1]
        _control = carla.VehicleControl()
        # trajectory_fig.set_invert_x_axis() # Because UE4 uses left-handed 
                                           # coordinate system the X
                                           # axis in the graph is flipped
        trajectory_fig.set_axis_equal()    # X-Y spacing should be equal in size

        # Add waypoint markers
        t = world.player.get_transform()
        start_x = t.location.x
        start_y = t.location.y
        INTERP_MAX_POINTS_PLOT = 20
        # Add trajectory markers
        trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                 x0=[start_x]*TOTAL_EPISODE_FRAMES, 
                                 y0=[start_y]*TOTAL_EPISODE_FRAMES,
                                 color=[1, 0.5, 0])
        # Add lookahead path
        trajectory_fig.add_graph("lookahead_path", 
                                 window_size=INTERP_MAX_POINTS_PLOT,
                                 x0=[start_x]*INTERP_MAX_POINTS_PLOT, 
                                 y0=[start_y]*INTERP_MAX_POINTS_PLOT,
                                 color=[0, 0.7, 0.7],
                                 linewidth=1)
        trajectory_fig.add_graph("ref_path", 
                                 window_size=N,
                                 x0=[start_x]*N, 
                                 y0=[start_y]*N,
                                 color=[0, 0.7, 0.7],
                                 linewidth=1)
        KP = 0.3
        curr_speed = 0
        all_vehicles = np.ones((max_no_of_vehicles,6))*10000
        if file_path_follow != None:
            trajectory_to_follow = np.loadtxt(file_path_follow,delimiter = ",")
        else :
            trajectory_to_follow=None
        trajectory_to_follow[:,1] = -trajectory_to_follow[:,1]
        if file_centre_line != None:
            centre_line = np.loadtxt(file_centre_line,delimiter = ",")
        else :
            centre_line=None
        traj_followed = []
        centre_line[:,1] = -centre_line[:,1]
        
        tx, ty, tyaw, tc, ts, csp = generate_target_course(trajectory_to_follow[:,0], trajectory_to_follow[:,1])
        tx_center, ty_center, tyaw_center, tc_center, ts_center, csp_center = generate_target_course(centre_line[:,0], centre_line[:,1])
        trajectory_to_follow = np.array([tx,ty,ts]).T
        print(trajectory_to_follow.shape)
        trajectory_fig.add_graph("waypoints", window_size=len(ty),
                                 x0=tx, y0=ty,
                                 linestyle="-", marker="", color='g')
        trajectory_fig.add_graph("left_boundary", window_size=len(ty_center),
                                 x0=tx_center-5*np.sin(tyaw_center), y0=ty_center+5*np.cos(tyaw_center),
                                 linestyle="-", marker="", color='r')
        trajectory_fig.add_graph("right_boundary", window_size=len(ty_center),
                                 x0=tx_center+5*np.sin(tyaw_center), y0=ty_center-5*np.cos(tyaw_center),
                                 linestyle="-", marker="", color='r')
        
        itr = 0
        total_itr=0
        curr_steering_array = np.zeros((N,2))
        target_speed_array = np.array([0]*N)
        trajectory_fig.add_graph("car", window_size=1, 
                                 marker="s", color='b', markertext="Car",
                                 marker_text_offset=1)
        trajectory_fig.add_graph("car_shifted", window_size=1, 
                                 marker="s", color='g', markertext="Car",
                                 marker_text_offset=1)
        _control = carla.VehicleControl()
        # player = random.choice(world.get_actors().filter('*mustang*'))
        speeds_time = []
        t = world.player.get_transform()
        angle_heading = t.rotation.yaw * pi/ 180
        world.player.set_velocity(carla.Vector3D(float(40*math.cos(angle_heading)),float(40*math.sin(angle_heading)),0))
        world.player.apply_control(carla.VehicleControl(throttle=0, brake=0))
        while True:
            _control.steer = 0
            _control.throttle = 1
            if itr > 400 :
                break
            itr += 1
            # world.player.apply_control(_control)
            world.tick(clock)
            client.get_world().tick()
            world.render(display)
            world.player.apply_control(_control)
            v = world.player.get_velocity()
            speeds_time.append([hud.simulation_time,math.sqrt(v.x**2 + v.y**2),world.imu_sensor.accelerometer[0],world.imu_sensor.accelerometer[1]])
            pygame.display.flip()

        store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
        # np.savetxt('outputs/comp_times.csv',np.array(time_estimates))
        np.savetxt('outputs/speed_comp.csv',np.array(speeds_time))
        

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()

CONTROLLER_OUTPUT_FOLDER = './outputs'

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
    """ Store the resulting plot.
    """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "model3")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)
    global pub
    pub = rospy.Publisher('chatter', custom_msg, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    
    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()

