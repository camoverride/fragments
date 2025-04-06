import pygame
import multiprocessing
import os
import subprocess
import signal

def get_monitor_geometry(name):
    """Get x position and resolution for monitor by name"""
    try:
        output = subprocess.check_output(['xrandr', '--query']).decode()
        for line in output.splitlines():
            if name in line and ' connected' in line:
                parts = line.split()
                res_pos = parts[2]  # e.g. 1600x900+1920+0
                width, height = map(int, res_pos.split('+')[0].split('x'))
                x_pos = int(res_pos.split('+')[1])
                return x_pos, width, height
    except Exception as e:
        print(f"Error getting monitor info: {e}")
    return None

def show_image(monitor_name, image_path):
    """Display image on specified monitor"""
    geo = get_monitor_geometry(monitor_name)
    if not geo:
        print(f"Monitor {monitor_name} not found!")
        return

    x_pos, width, height = geo
    
    # Set window position before init
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_pos},0"
    os.environ['SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR'] = "0"
    
    pygame.init()
    try:
        # Create borderless fullscreen window
        screen = pygame.display.set_mode(
            (width, height),
            pygame.NOFRAME | pygame.HWSURFACE
        )
        
        # Hide mouse cursor
        pygame.mouse.set_visible(False)
        
        # Load and scale image
        img = pygame.transform.scale(
            pygame.image.load(image_path),
            (width, height)
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
            
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()

if __name__ == "__main__":
    # Configure monitors by name
    config = {
        "DP-1": "face_1.jpg",
        "DVI-D-0": "face_2.jpg"
    }
    
    # Start processes
    processes = []
    for name, image in config.items():
        p = multiprocessing.Process(
            target=show_image,
            args=(name, image)
        )
        p.start()
        processes.append(p)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        for p in processes:
            p.terminate()
        os._exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Wait for processes
    for p in processes:
        p.join()