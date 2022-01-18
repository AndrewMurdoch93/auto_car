from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import cubic_spline_planner
import yaml

def generate_circle_image():
    from matplotlib import image
    image = Image.new('RGBA', (600, 600))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 600, 600), fill = 'black', outline ='black')
    draw.ellipse((50, 50, 550, 550), fill = 'white', outline ='white')
    draw.ellipse((150, 150, 450, 450), fill = 'black', outline ='black')
    draw.point((100, 100), 'red')
    image_path = sys.path[0] + '\\maps\\circle' + '.png'
    image.save(image_path, 'png')

def generate_circle_goals():
    from matplotlib import image
    image_path = sys.path[0] + '\\maps\\circle' + '.png'
    im = image.imread(image_path)
    plt.imshow(im, extent=(0,30,0,30))

    R=10
    theta=np.linspace(0, 2*math.pi, 17)
    x = 15+R*np.cos(theta-math.pi/2)
    y = 15+R*np.sin(theta-math.pi/2)
    rx, ry, ryaw, rk, s = cubic_spline_planner.calc_spline_course(x, y)
    #plt.plot(rx, ry, "-r", label="spline")
    #plt.plot(x, y, 'x')
    #plt.show()
    return x, y, rx, ry, ryaw, rk, s

#enerate_circle_image()
#generate_circle_goals()
