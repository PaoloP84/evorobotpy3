VIEWPORT_W = 960
VIEWPORT_H = 720

SCALE = 1.0

START_W = 150 / SCALE
START_H = 50 / SCALE

screen = None
clock = None

def setScale(scale):
    global SCALE
    global START_W
    global START_H
    
    SCALE = scale
    # Update start width and height
    START_W = 150 / SCALE
    START_H = 50 / SCALE
    
def setWindow(width, height):
    global VIEWPORT_W
    global VIEWPORT_H
    
    VIEWPORT_W = width
    VIEWPORT_H = height

def update(wobj, info, ob, ac, nact):
    global VIEWPORT_W
    global VIEWPORT_H
    global SCALE
    global START_W
    global START_H
    global screen
    global clock
    
    try:
       import pygame
       from pygame import gfxdraw
    except ImportError as e:
       raise DependencyNotInstalled(
            "pygame is not installed, run `pip install gymnasium[box2d]`"
       ) from e
       
    if screen is None:
        pygame.init()
        pygame.display.init()
        screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
    if clock is None:
        clock = pygame.time.Clock()

    surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

    pygame.transform.scale(surf, (SCALE, SCALE))

    pygame.draw.polygon(
        surf,
        color=(255, 255, 255),
        points=[
            (0, 0),
            (VIEWPORT_W, 0),
            (VIEWPORT_W, VIEWPORT_H),
            (0, VIEWPORT_H),
        ],
    )
    
    # Draw objects
    c = 0
    nobjects = 0
    
    while (wobj[c] > 0.0):
        # circular robot
        if (wobj[c] == 1.0):
            ccolor = (int(wobj[c+5]),int(wobj[c+6]),int(wobj[c+7]))
            filled = True
            if ccolor[0] < 0.0:
                ccolor = (0,0,0)
                filled = False
            # Draw half circle representing LEDs
            if filled:
                pygame.draw.circle(
                    surf,
                    color=ccolor,
                    center=(START_W + wobj[c+1] / SCALE, START_H + wobj[c+2] / SCALE),
                    radius=wobj[c+3] / SCALE,
                )
            else:
                pygame.draw.circle(
                    surf,
                    color=ccolor,
                    center=(START_W + wobj[c+1] / SCALE, START_H + wobj[c+2] / SCALE),
                    radius=wobj[c+3] / SCALE,
                    width=1,
                )
            # Draw orientation of the agent
            pygame.draw.aaline(
                surf,
                start_pos=(START_W + wobj[c+1] / SCALE, START_H + wobj[c+2] / SCALE),
                end_pos=(START_W + wobj[c+8] / SCALE, START_H + wobj[c+9] / SCALE),
                color=(0, 0, 0),
            )
        # line
        if (wobj[c] == 2.0):
            ccolor = (int(wobj[c+5]),int(wobj[c+6]),int(wobj[c+7]))
            pygame.draw.aaline(
                surf,
                start_pos=(START_W + wobj[c+1] / SCALE, START_H + wobj[c+2] / SCALE),
                end_pos=(START_W + wobj[c+3] / SCALE, START_H + wobj[c+4] / SCALE),
                color=ccolor,
            )
        # circle
        if (wobj[c] == 3.0):
            ccolor = (int(wobj[c+5]),int(wobj[c+6]),int(wobj[c+7]))
            pygame.draw.circle(
                surf,
                color=ccolor,
                center=(START_W + wobj[c+1] / SCALE, START_H + wobj[c+2] / SCALE),
                radius=wobj[c+3] / SCALE,
            )
        # polygon
        if (wobj[c] == 4.0):
            ccolor = (int(wobj[c+5]),int(wobj[c+6]),int(wobj[c+7]))
            path = [(START_W + wobj[c+1] / SCALE, START_H + wobj[c+2] / SCALE), (START_W + wobj[c+3] / SCALE, START_H + wobj[c+2] / SCALE), (START_W + wobj[c+3] / SCALE, START_H + wobj[c+4] / SCALE), (START_W + wobj[c+1] / SCALE, START_H + wobj[c+4] / SCALE)]
            pygame.draw.polygon(surf, color=ccolor, points=path)
            gfxdraw.aapolygon(surf, path, ccolor)
        c = c + 10
        nobjects += 1

    surf = pygame.transform.flip(surf, False, True)

    assert screen is not None
    screen.blit(surf, (0, 0))
    pygame.event.pump()
    clock.tick(30)
    pygame.display.flip()
        
