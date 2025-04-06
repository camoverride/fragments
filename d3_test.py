import pygame
import os
import subprocess

# Monitor config from your xrandr output
MONITORS = {
    "DVI-D-0": {"x": 0,    "width": 1600, "height": 900, "image": "face_1.jpg"},
    "DP-1":    {"x": 1600, "width": 1600, "height": 900, "image": "face_2.jpg"}, 
    "HDMI-0":  {"x": 3200, "width": 1920, "height": 1080, "image": "face_1.jpg"}
}

def launch_display(monitor_name):
    """Launch one display in its own process"""
    m = MONITORS[monitor_name]
    
    # NVIDIA-specific optimizations
    os.environ["__GL_SYNC_DISPLAY_DEVICE"] = monitor_name
    os.environ["DISPLAY"] = ":0"  # Force primary X display
    
    # Position and create window
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{m['x']},0"
    pygame.init()
    screen = pygame.display.set_mode(
        (m['width'], m['height']), 
        pygame.NOFRAME | pygame.HWSURFACE
    )
    
    # Load and display image
    img = pygame.image.load(m['image'])
    img = pygame.transform.scale(img, (m['width'], m['height']))
    screen.blit(img, (0, 0))
    pygame.display.flip()
    
    # Keep window open without looping
    while True:
        pygame.time.wait(1000)  # Minimal CPU usage

if __name__ == "__main__":
    # Start each monitor in separate process
    import multiprocessing
    for monitor in MONITORS:
        multiprocessing.Process(
            target=launch_display,
            args=(monitor,),
            daemon=True
        ).start()
    
    # Keep main process alive
    while True:
        pass