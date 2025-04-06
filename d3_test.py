import pygame
import multiprocessing
import os

MONITOR_SETUP = {
    "DP-1": {"x": 1920, "width": 1600, "height": 900},
    "DVI-D-0": {"x": 3520, "width": 1600, "height": 900}
}

def display_image(monitor_name, image_path):
    m = MONITOR_SETUP[monitor_name]
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{m['x']},0"
    
    pygame.init()
    # Borderless window sized to monitor dimensions
    screen = pygame.display.set_mode(
        (m['width'], m['height']),
        pygame.NOFRAME | pygame.HWSURFACE
    )
    
    # Hide mouse cursor
    pygame.mouse.set_visible(False)
    
    img = pygame.transform.scale(pygame.image.load(image_path), (m['width'], m['height']))
    
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
    processes = [
        multiprocessing.Process(target=display_image, args=("DP-1", "face_1.jpg")),
        multiprocessing.Process(target=display_image, args=("DVI-D-0", "face_2.jpg"))
    ]
    
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()