import pygame
import numpy as np

SIZE = 40

class Display:
    
    def __init__(self, state_shape):
        pygame.init()
        pygame.event.pump()
        self.surface = pygame.display.set_mode([SIZE*state_shape[1], SIZE*state_shape[0]])
        self.state_shape = state_shape

        self.head_up = pygame.image.load('snake_Graphics/head_up.png').convert_alpha()
        self.head_right = pygame.image.load('snake_Graphics/head_right.png').convert_alpha()
        self.head_down = pygame.image.load('snake_Graphics/head_down.png').convert_alpha()
        self.head_left = pygame.image.load('snake_Graphics/head_left.png').convert_alpha()

        self.tail_up = pygame.image.load('snake_Graphics/tail_up.png').convert_alpha()
        self.tail_right = pygame.image.load('snake_Graphics/tail_right.png').convert_alpha()
        self.tail_down = pygame.image.load('snake_Graphics/tail_down.png').convert_alpha()
        self.tail_left = pygame.image.load('snake_Graphics/tail_left.png').convert_alpha()

        self.body_vertical = pygame.image.load('snake_Graphics/body_vertical.png').convert_alpha()
        self.body_horizontal = pygame.image.load('snake_Graphics/body_horizontal.png').convert_alpha()

        self.body_tr = pygame.image.load('snake_Graphics/body_topright.png').convert_alpha()
        self.body_tl = pygame.image.load('snake_Graphics/body_topleft.png').convert_alpha()
        self.body_br = pygame.image.load('snake_Graphics/body_bottomright.png').convert_alpha()
        self.body_bl = pygame.image.load('snake_Graphics/body_bottomleft.png').convert_alpha()

        self.apple = pygame.image.load('snake_Graphics/apple.png').convert_alpha()

        self.dir_ind = {
            0 : self.head_up,
            1 : self.head_right,
            2 : self.head_down,
            3 : self.head_left
        }
        
        self.body_display = {
            (1, 0, -1, 0) : self.body_vertical,
            (-1, 0, 1, 0) : self.body_vertical,
            (0, 1, 0, -1) : self.body_horizontal,
            (0, -1, 0, 1) : self.body_horizontal,
            (-1, 0, 0, 1) : self.body_bl,
            (0, 1, -1, 0) : self.body_bl,
            (0, -1, -1, 0) : self.body_br,
            (-1, 0, 0, -1) : self.body_br,
            (0, 1, 1, 0) : self.body_tl,
            (1, 0, 0, 1) : self.body_tl,
            (0, -1, 1, 0) : self.body_tr,
            (1, 0, 0, -1) : self.body_tr
        }

        self.tail_display = {
            (1,0) : self.tail_up,
            (0, -1) : self.tail_right,
            (-1, 0) : self.tail_down,
            (0, 1) : self.tail_left
        }


    def refresh(self, apple_pos, snake_body, curr_dir_ind, score, episode_nm):
        pygame.event.pump()

        self.surface.fill((168, 242, 60))
        rect = pygame.Rect(apple_pos[1] * SIZE, apple_pos[0]*SIZE, SIZE, SIZE)
        self.surface.blit(self.apple, rect)

        for i in range(1, len(snake_body) - 1):
            rect = pygame.Rect(snake_body[i][1]*SIZE, snake_body[i][0]*SIZE, SIZE, SIZE)
            diff_1 = tuple(np.asarray(snake_body[i]) - np.asarray(snake_body[i-1]))
            diff_2 = tuple(np.asarray(snake_body[i]) - np.asarray(snake_body[i+1]))

            self.surface.blit(self.body_display[diff_1 + diff_2], rect)
        
        rect = pygame.Rect(snake_body[-1][1]*SIZE, snake_body[-1][0]*SIZE, SIZE, SIZE)
        tail_diff = tuple(np.asarray(snake_body[-2]) - np.asarray(snake_body[-1]))
        self.surface.blit(self.tail_display[tail_diff], rect)

        rect = pygame.Rect(snake_body[0][1]*SIZE, snake_body[0][0]*SIZE, SIZE, SIZE)
        self.surface.blit(self.dir_ind[curr_dir_ind], rect)

        pygame.draw.rect(self.surface, (170,170,170), [0, 0, SIZE*self.state_shape[1], SIZE])
        pygame.draw.rect(self.surface, (170,170,170), [0, 0, SIZE, SIZE*self.state_shape[0]])
        pygame.draw.rect(self.surface, (170,170,170), [(self.state_shape[1] - 1)*SIZE, 0, SIZE, SIZE*self.state_shape[0]])
        pygame.draw.rect(self.surface, (170,170,170), [0, (self.state_shape[0] - 1)*SIZE, SIZE*self.state_shape[1] ,SIZE])

        title = "Snake, episode: {}, score: {}".format(episode_nm, score)
        pygame.display.set_caption(title)
        pygame.display.flip()

    