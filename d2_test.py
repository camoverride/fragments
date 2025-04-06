import pygame
import multiprocessing
import os



def show_image(display_num, resolution, image_path):
    """
    Display one image fullscreen on one monitor
    """
    os.environ["DISPLAY"] = f":0.{display_num}"
    pygame.init()
    screen = pygame.display.set_mode(resolution, pygame.FULLSCREEN, display=display_num)
    img = pygame.transform.scale(pygame.image.load(image_path), resolution)

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
    # Define monitor 1 (DP-1)
    p1 = multiprocessing.Process(
        target=show_image,
        args=(0, (1600, 900), "face_1.jpg")  # display_num, resolution, image
    )

    # Define monitor 2 (DVI-D-0)
    p2 = multiprocessing.Process(
        target=show_image,
        args=(1, (1600, 900), "face_2.jpg")
    )

    # Start both processes
    p1.start()
    p2.start()

    # Wait for both to finish
    p1.join()
    p2.join()
