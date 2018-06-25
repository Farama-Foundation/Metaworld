from enum import Enum

import pygame
from collections import Iterable


class PygameViewer(object):
    def __init__(
            self,
            screen_width=640,
            screen_height=480,
            x_bounds=(0, 640),
            y_bounds=(0, 480),
            render_onscreen=True,
    ):
        """
        All xy-coordinates are scaled linear to map from
            x_bounds --> [0, screen_width-1]
        and similarly for y.

        Width and heights are also scaled. For radius, the min of the x-scale
        and y-scale is taken.

        :param screen_width:
        :param screen_height:
        :param x_bounds:
        :param y_bounds:
        """
        self.width = screen_width
        self.height = screen_width
        self.x_scaler = LinearMapper(x_bounds, (0, screen_width - 1))
        self.y_scaler = LinearMapper(y_bounds, (0, screen_height - 1))
        self.terminated = False
        self.clock = pygame.time.Clock()
        self.render_onscreen = render_onscreen
        if self.render_onscreen:
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        else:
            self.screen = pygame.Surface((screen_width, screen_height))

    def render(self):
        if self.render_onscreen:
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.terminated = True

    def fill(self, color):
        self.screen.fill(color)

    def tick(self, dt):
        self.clock.tick(dt)

    def draw_segment(self, p1, p2, color):
        p1 = self.convert_xy(p1)
        p2 = self.convert_xy(p2)
        pygame.draw.aaline(self.screen, color, p1, p2)

    def draw_circle(self, center, radius, color, thickness=1):
        center = self.convert_xy(center)
        radius = self.scale_min(radius)

        pygame.draw.circle(self.screen, color, center, radius, thickness)

    def draw_solid_circle(self, center, radius, color):
        self.draw_circle(center, radius, color, thickness=0)

    def draw_rect(self, point, width, height, color, thickness):
        x, y = self.convert_xy(point)
        width = self.scale_x(width)
        height = self.scale_y(height)
        pygame.draw.rect(self.screen, color, (x, y, width, height), thickness)

    def convert_xy(self, point):
        x, y = point
        return int(self.x_scaler.convert(x)), int(self.y_scaler.convert(y))

    def scale_x(self, x):
        return int(self.x_scaler.scale(x))

    def scale_y(self, y):
        return int(self.y_scaler.scale(y))

    def scale_min(self, value):
        return min(self.scale_y(value), self.scale_y(value))

    def get_image(self):
        # s = pygame.display.get_surface()
        return pygame.surfarray.array3d(self.screen)

    def reinit_screen(self, render_onscreen):
        self.render_onscreen = render_onscreen
        if self.render_onscreen:
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = pygame.Surface((self.width, self.height))


class LinearMapper(object):
    """
    Convert a range
        [a, b] --> [c, d]
    with a linear mapping.

    Also supports just scaling a value.
    """
    def __init__(self, in_bounds, out_bounds):
        self.in_min, in_max = in_bounds
        self.out_min, out_max = out_bounds
        self.in_range = in_max - self.in_min
        self.out_range = out_max - self.out_min

    def convert(self, value):
        return (
            (((value - self.in_min) * self.out_range) / self.in_range)
            + self.out_min
        )

    def scale(self, value):
        return value * self.out_range / self.in_range
