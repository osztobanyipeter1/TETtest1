import pygame
from pygame.locals import DOUBLEBUF, OPENGL

pygame.init()
pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
print("OpenGL context létrejött!")
