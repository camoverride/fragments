import pygame
import subprocess
import os
import time

# Get exact position and resolution for a monitor by name
def get_monitor_info(name):
    try:
        output = subprocess.check_output(['xrandr', '--query']).decode()
        for line in output.splitlines():
            if name in line and ' connected' in line:
                parts = line.split()
                geometry = parts[2 if parts[2][0].isdigit() else 3]
                if '+' in geometry:
                    res, xpos, _ = geometry.split('+')
                    width, height = map(int, res.split('x'))
                    return int(xpos), width, height
    except Exception as e:
        print(f"Error getting monitor info: {e}")
    return None

def create_window(monitor_name, image_path):
    info = get_monitor_info(monitor_name)
    if not info:
        print(f"Could not get info for {monitor_name}")
        return
    
    xpos, width, height = info
    
    # Critical X11 settings for proper fullscreen
    os.environ['SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR'] = '0'
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{xpos},0"
    
    pygame.init()
    try:
        # Create true borderless fullscreen window
        screen = pygame.display.set_mode(
            (width, height),
            pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        
        # Hide mouse cursor
        pygame.mouse.set_visible(False)
        
        # Load and display image
        img = pygame.image.load(image_path)
        img = pygame.transform.scale(img, (width, height))
        
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            
            screen.blit(img, (0, 0))
            pygame.display.flip()
            clock.tick(30)
            
    finally:
        pygame.quit()

if __name__ == "__main__":
    # Configuration - edit these to match your setup
    MONITORS = {
        "DP-1": "face_1.jpg",
        "DVI-D-0": "face_2.jpg"
    }
    
    # Start each window in a separate process
    processes = []
    for name, image in MONITORS.items():
        p = subprocess.Popen([
            'python3', '-c',
            f"import pygame, os; os.environ['SDL_VIDEO_WINDOW_POS']='{get_monitor_info(name)[0]},0'; "
            f"pygame.init(); screen=pygame.display.set_mode({get_monitor_info(name)[1:]}, pygame.NOFRAME); "
            f"img=pygame.transform.scale(pygame.image.load('{image}'), {get_monitor_info(name)[1:]}); "
            "clock=pygame.time.Clock(); "
            "running=True; "
            "while running: "
            "    for e in pygame.event.get(): "
            "        if e.type==pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE): running=False; "
            "    screen.blit(img, (0,0)); pygame.display.flip(); clock.tick(30); "
            "pygame.quit()"
        ])
        processes.append(p)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()