import pygame
import subprocess

# Initialize Pygame
pygame.init()

# Load images
face1 = pygame.image.load('face_1.jpg')
face2 = pygame.image.load('face_2.jpg')

# Get monitor positions using xrandr (Linux specific)
def get_monitor_positions():
    output = subprocess.check_output(['xrandr']).decode()
    positions = {}
    for line in output.splitlines():
        if ' connected' in line:
            parts = line.split()
            name = parts[0]
            if '+' in parts[2]:  # e.g., 1600x900+1920+0
                x_pos = int(parts[2].split('+')[1])
                positions[name] = x_pos
    return positions

monitor_pos = get_monitor_positions()

# Create one large virtual display spanning all monitors
total_width = 5120  # From your xrandr output (1600 + 1920 + 1600)
screen = pygame.display.set_mode((total_width, 900))

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
    
    # Display images at correct positions
    screen.blit(pygame.transform.scale(face1, (1600, 900)), (monitor_pos['DP-1'], 0))
    screen.blit(pygame.transform.scale(face2, (1600, 900)), (monitor_pos['DVI-D-0'], 0))
    pygame.display.flip()

pygame.quit()