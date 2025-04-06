import pygame
import os
import multiprocessing
import numpy as np



# Monitor configurations from your xrandr output
DVI_D_0 = {"x": 0,    "width": 1600, "height": 900, "image": "face_1.jpg"}
DP_1    = {"x": 1600, "width": 1600, "height": 900, "image": "face_2.jpg"}
HDMI_0  = {"x": 3200, "width": 1920, "height": 1080, "image": "face_1.jpg"}

def display_dvi_d_0():
    os.environ["__GL_SYNC_DISPLAY_DEVICE"] = "DVI-D-0"
    os.environ["DISPLAY"] = ":0"
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{DVI_D_0['x']},0"
    
    pygame.init()
    screen = pygame.display.set_mode((DVI_D_0['width'], DVI_D_0['height']), pygame.NOFRAME | pygame.HWSURFACE)

    img_path = DVI_D_0['image']
    last_mtime = 0
    img_surface = None

    while True:
        try:
            current_mtime = os.path.getmtime(img_path)
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                img = pygame.image.load(img_path)
                img = pygame.transform.rotate(img, 90)
                img = pygame.transform.scale(img, (DVI_D_0['width'], DVI_D_0['height']))
                img_surface = img
            if img_surface:
                screen.blit(img_surface, (0, 0))
                pygame.display.flip()
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
        
        pygame.time.wait(1000)

def display_dp_1():
    os.environ["__GL_SYNC_DISPLAY_DEVICE"] = "DP-1"
    os.environ["DISPLAY"] = ":0"
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{DP_1['x']},0"
    
    pygame.init()
    screen = pygame.display.set_mode((DP_1['width'], DP_1['height']), pygame.NOFRAME | pygame.HWSURFACE)

    img_path = DP_1['image']
    last_mtime = 0
    img_surface = None

    while True:
        try:
            current_mtime = os.path.getmtime(img_path)
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                img = pygame.image.load(img_path)
                img = pygame.transform.rotate(img, -90)
                img = pygame.transform.scale(img, (DP_1['width'], DP_1['height']))
                img_surface = img
            if img_surface:
                screen.blit(img_surface, (0, 0))
                pygame.display.flip()
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
        
        pygame.time.wait(1000)
def display_hdmi_0():
    os.environ["__GL_SYNC_DISPLAY_DEVICE"] = "HDMI-0"
    os.environ["DISPLAY"] = ":0"
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{HDMI_0['x']},0"
    
    pygame.init()
    screen = pygame.display.set_mode((HDMI_0['width'], HDMI_0['height']), pygame.NOFRAME | pygame.HWSURFACE)

    npz_path = 'animation.npz'
    last_mtime = 0
    frames = []
    current_frame = 0
    clock = pygame.time.Clock()

    while True:
        try:
            current_mtime = os.path.getmtime(npz_path)
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                frames = np.load(npz_path)['frames']
                current_frame = 0  # Reset to beginning

            if frames is not None and len(frames) > 0:
                frame = frames[current_frame]
                if len(frame.shape) == 2:
                    frame = np.stack((frame,) * 3, axis=-1)
                frame = frame[..., ::-1]  # RGB to BGR
                frame = np.flipud(frame)
                surf = pygame.surfarray.make_surface(frame)
                surf = pygame.transform.scale(surf, (HDMI_0['width'], HDMI_0['height']))
                screen.blit(surf, (0, 0))
                pygame.display.flip()

                current_frame = (current_frame + 1) % len(frames)
        except Exception as e:
            print(f"Error loading animation {npz_path}: {e}")
        
        clock.tick(30)





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