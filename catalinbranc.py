import pygame, sys

pygame.init()
#setare rez ecran
screen = pygame.display.set_mode((1280, 720))

#declaratie variabila de tip clock
clock = pygame.time.Clock()

bg_surface = pygame.image.load('8650883-wallpaper.jpg') 

pasare_surface = pygame.image.load('Flappy-Bird-PNG-Background.png').convert_alpha()
pasare_surface = pygame.transform.scale(pasare_surface, (100, 100))
pasare_rect = pasare_surface.get_rect(center = (100, 100))

gravity = 0.15
movement_p = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                movement_p -= 10
                print("xfd")
        #iesire pygame + system
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    #draw pe ecran
    screen.blit(bg_surface, (0,0))
    movement_p += gravity
    pasare_rect.centery += movement_p
    screen.blit(pasare_surface, pasare_rect)
    #refresh display joc      
    
        
    pygame.display.update()
    
    #initializare fps in functie de timp
    clock.tick(60)
    
    
    
    
    

    
    
    