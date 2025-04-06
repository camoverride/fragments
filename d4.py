import pygame
import os
import multiprocessing
import numpy as np



# Monitor configurations from your xrandr output
DVI_D_0 = {"x": 0,    "width": 1600, "height": 900, "image": "face_1.jpg"}
DP_1    = {"x": 1600, "width": 1600, "height": 900, "image": "face_2.jpg"}
HDMI_0  = {"x": 3200, "width": 1920, "height": 1080, "image": "face_1.jpg"}

def display_dvi_d_0():
    """Display on DVI-D-0 monitor"""
    os.environ["__GL_SYNC_DISPLAY_DEVICE"] = "DVI-D-0"
    os.environ["DISPLAY"] = ":0"
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{DVI_D_0['x']},0"
    
    pygame.init()
    screen = pygame.display.set_mode(
        (DVI_D_0['width'], DVI_D_0['height']), 
        pygame.NOFRAME | pygame.HWSURFACE
    )
    
    img = pygame.image.load(DVI_D_0['image'])
    img = np.rot90(img)
    img = pygame.transform.scale(img, (DVI_D_0['width'], DVI_D_0['height']))
    screen.blit(img, (0, 0))
    pygame.display.flip()
    
    while True:
        pygame.time.wait(1000)

def display_dp_1():
    """Display on DP-1 monitor"""
    os.environ["__GL_SYNC_DISPLAY_DEVICE"] = "DP-1"
    os.environ["DISPLAY"] = ":0"
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{DP_1['x']},0"
    
    pygame.init()
    screen = pygame.display.set_mode(
        (DP_1['width'], DP_1['height']), 
        pygame.NOFRAME | pygame.HWSURFACE
    )
    
    img = pygame.image.load(DP_1['image'])
    img = pygame.transform.scale(img, (DP_1['width'], DP_1['height']))
    screen.blit(img, (0, 0))
    pygame.display.flip()
    
    while True:
        pygame.time.wait(1000)

# def display_hdmi_0():
#     """Display on HDMI-0 monitor"""
#     os.environ["__GL_SYNC_DISPLAY_DEVICE"] = "HDMI-0"
#     os.environ["DISPLAY"] = ":0"
#     os.environ['SDL_VIDEO_WINDOW_POS'] = f"{HDMI_0['x']},0"
    
#     pygame.init()
#     screen = pygame.display.set_mode(
#         (HDMI_0['width'], HDMI_0['height']), 
#         pygame.NOFRAME | pygame.HWSURFACE
#     )
    
#     img = pygame.image.load(HDMI_0['image'])
#     img = pygame.transform.scale(img, (HDMI_0['width'], HDMI_0['height']))
#     screen.blit(img, (0, 0))
#     pygame.display.flip()
    
#     while True:
#         pygame.time.wait(1000)


def display_hdmi_0():
    """Play animation on HDMI-0 monitor"""
    os.environ["__GL_SYNC_DISPLAY_DEVICE"] = "HDMI-0"
    os.environ["DISPLAY"] = ":0"
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{HDMI_0['x']},0"
    
    pygame.init()
    screen = pygame.display.set_mode(
        (HDMI_0['width'], HDMI_0['height']), 
        pygame.NOFRAME | pygame.HWSURFACE
    )
    
    # Load frames from NPZ file
    frames = np.load('animation.npz')['frames']
    
    clock = pygame.time.Clock()
    current_frame = 0
    
    while True:
        # Get current frame and display it
        frame = frames[current_frame]
        if len(frame.shape) == 2:  # Grayscale to RGB
            frame = np.stack((frame,)*3, axis=-1)
        frame = frame[..., ::-1]  # RGB to BGR (fixes blue tint)
        
        surf = pygame.surfarray.make_surface(frame)
        surf = pygame.transform.scale(surf, (HDMI_0['width'], HDMI_0['height']))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Move to next frame (loop if needed)
        current_frame = (current_frame + 1) % len(frames)
        clock.tick(30)  # 30 FPS playback




if __name__ == "__main__":
    # Start all three displays as separate processes
    p1 = multiprocessing.Process(target=display_dvi_d_0, daemon=True)
    p2 = multiprocessing.Process(target=display_dp_1, daemon=True)
    p3 = multiprocessing.Process(target=display_hdmi_0, daemon=True)
    
    p1.start()
    p2.start()
    p3.start()
    
    # Keep main process alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        p1.terminate()
        p2.terminate()
        p3.terminate()