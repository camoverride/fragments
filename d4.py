import pygame
import os
import subprocess

# Monitor config from your xrandr output
MONITORS = {
    "DVI-D-0": {"x": 0,    "width": 1600, "height": 900, "image": "face_1.jpg"},
    "DP-1":    {"x": 1600, "width": 1600, "height": 900, "image": "face_2.jpg"}, 
    "HDMI-0":  {"x": 3200, "width": 1920, "height": 1080, "image": "face_1.jpg"}
}



"""Launch one display in its own process"""

# NVIDIA-specific optimizations
os.environ["__GL_SYNC_DISPLAY_DEVICE"] = "DVI-D-0"
os.environ["DISPLAY"] = ":0"  # Force primary X display

# Position and create window
os.environ['SDL_VIDEO_WINDOW_POS'] = f"0,0"
pygame.init()
screen = pygame.display.set_mode(
    (1600, 900), 
    pygame.NOFRAME | pygame.HWSURFACE
)

# Load and display image
img = pygame.image.load("face_1.jpg")
img = pygame.transform.scale(img, (1600, 900))
screen.blit(img, (0, 0))
pygame.display.flip()

# Keep window open without looping
while True:
    pygame.time.wait(1000)  # Minimal CPU usage
