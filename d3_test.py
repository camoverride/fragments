import pygame
import multiprocessing
import os
import subprocess

# Monitor configurations - adjust these to match your xrandr output
MONITORS = {
    "DP-1": {"display": 0, "width": 1600, "height": 900},
    "DVI-D-0": {"display": 1, "width": 1600, "height": 900}
}

def show_image(monitor_name, image_path):
    """Display image on specified monitor in true fullscreen"""
    m = MONITORS[monitor_name]
    
    # Set up environment for this display
    os.environ['DISPLAY'] = f":{m['display']}"
    os.environ['SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR'] = "0"
    
    pygame.init()
    
    # Get available displays
    display_info = pygame.display.Info()
    print(f"Initializing {monitor_name} on display {m['display']}")
    
    try:
        # Create fullscreen window
        screen = pygame.display.set_mode(
            (m['width'], m['height']),
            pygame.FULLSCREEN | pygame.NOFRAME,
            display=m['display']
        )
        
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
            
    except Exception as e:
        print(f"Error on {monitor_name}: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    # Start a process for each monitor
    processes = [
        multiprocessing.Process(target=show_image, args=("DP-1", "face_1.jpg")),
        multiprocessing.Process(target=show_image, args=("DVI-D-0", "face_2.jpg"))
    ]
    
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()