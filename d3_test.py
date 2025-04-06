import pygame
import multiprocessing
import os
import subprocess

# Force fullscreen settings (disable window decorations)
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
os.environ['SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR'] = "0"  # Disable compositor
os.environ['SDL_VIDEO_X11_FORCE_EGL'] = "1"  # Force hardware acceleration

# Monitor configurations from xrandr
MONITORS = {
    "DP-1": {"x": 1920, "width": 1600, "height": 900},
    "DVI-D-0": {"x": 3520, "width": 1600, "height": 900}
}

def hide_system_bars():
    """Hide Ubuntu dock and top bar"""
    subprocess.run(["gsettings", "set", "org.gnome.shell.extensions.dash-to-dock", "autohide", "true"])
    subprocess.run(["gsettings", "set", "org.gnome.shell.extensions.dash-to-dock", "dock-fixed", "false"])

def show_image(monitor_name, image_path):
    """Fullscreen display on specified monitor"""
    m = MONITORS[monitor_name]
    
    # Position window first
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{m['x']},0"
    
    pygame.init()
    
    # Create undecorated fullscreen window
    screen = pygame.display.set_mode(
        (m['width'], m['height']),
        pygame.FULLSCREEN | pygame.NOFRAME | pygame.HWSURFACE,
        display=0  # Will be positioned via SDL_VIDEO_WINDOW_POS
    )
    
    # Hide mouse cursor
    pygame.mouse.set_visible(False)
    
    # Load and scale image
    img = pygame.transform.scale(
        pygame.image.load(image_path),
        (m['width'], m['height'])
    )
    
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        screen.blit(img, (0, 0))
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    hide_system_bars()  # Hide Ubuntu UI elements
    
    processes = [
        multiprocessing.Process(target=show_image, args=("DP-1", "face_1.jpg")),
        multiprocessing.Process(target=show_image, args=("DVI-D-0", "face_2.jpg"))
    ]
    
    for p in processes:
        p.start()
    
    try:
        for p in processes:
            p.join()
    finally:
        # Restore system UI when done
        subprocess.run(["gsettings", "set", "org.gnome.shell.extensions.dash-to-dock", "autohide", "false"])
        subprocess.run(["gsettings", "set", "org.gnome.shell.extensions.dash-to-dock", "dock-fixed", "true"])