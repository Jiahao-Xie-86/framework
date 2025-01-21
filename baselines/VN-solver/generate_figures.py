import numpy as np
import os
import random
# Save the image as a file (optional)
import imageio
from PIL import Image

horizontal = 1
# Vertical indicates b parameter in ellipse layout.
# Circular layout is a special case of ellipse layout when vertical=1.
vertical = 1
# resolution indicates resolution parameter in spiral layout.
resolution = 0.35
# coloring = ['gray', 'random', 'uniform']
coloring = 'uniform'
# visualization = ['ellipse', 'random', 'spiral']
visualization = 'ellipse'
# model_size = ['small', 'medium']
model_size = 'small'





random.seed(23)
np.random.seed(23)

if model_size == 'medium':
    min_node = 20
    max_node = 50
if model_size == 'small':
    min_node = 4
    max_node = 20


def ellipse_layout(number_of_nodes, horizontal=1, vertical=1, height_pixels=224, width_pixels=224, scale=100, center=(112, 112), dim=2):
    #node_indices = [i for i in range(number_of_nodes)]
    paddims = max(0, (dim - 2))
    if number_of_nodes == 0:
        pos = {}
    elif number_of_nodes == 1:
        pos = [height_pixels//2, width_pixels//2]
    else:
        # Discard the extra angle since it matches 0 radians.
        theta = np.linspace(0, 1, number_of_nodes + 1)[:-1] * 2 * np.pi
        theta = theta.astype(np.float32)
        pos = np.column_stack(
            [horizontal * np.cos(theta), vertical * np.sin(theta), np.zeros((number_of_nodes, paddims))]
        )

        lim = 0  # max coordinate for all axes
        #for i in range(pos.shape[1]):
            #pos[:, i] -= pos[:, i].mean()
            #lim = max(abs(pos[:, i]).max(), lim)
        # rescale to (-scale, scale) in all directions, preserves aspect
        #if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale
        for i in range(pos.shape[1]):
            pos[:, i] = pos[:, i] + center[i]

        int_pos_x = []
        int_pos_y = []
        for each in pos:
            int_pos_x.append(round(each[0]))
            int_pos_y.append(round(each[1]))

        #pos = rescale_layout(pos, scale=scale) + center
        #pos = dict(zip(G, pos))

    return int_pos_x, int_pos_y


visualization = coloring + '_color_' + visualization

if 'ellipse' in visualization:
    visualization = str(vertical).replace('.', 'p') + '_' + visualization

if 'spiral' in visualization:
    visualization = str(resolution).replace('.', 'p') + '_sp_' + visualization

"""def draw_line(image, x1, y1, x2, y2, color):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    err = dx - dy

    while True:
        image[x1, y1] = color

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy"""

print(visualization)
id_num = -1
cycles = ['non_hamiltonian', 'hamiltonian']
for each in cycles:
    graph_files = [file for file in os.listdir(each) if file.endswith('.mat')]
    for file in graph_files:
        with open(str(each)+'/'+str(file), 'r') as f:
            matrices_text = f.read().strip().split('\n\n')

        graphs = []
        matrixs = []
        for matrix_text in matrices_text:
            if '1' in matrix_text:
                count = matrix_text.count('\n')
                if count >= min_node and count < max_node:
                    lines = matrix_text.strip().split('\n')
                    id_num+=1
                    matrix = np.loadtxt(lines, dtype=int)
                    matrixs.append(matrix)
                    x, y = ellipse_layout(len(matrix), horizontal=horizontal, vertical=vertical, height_pixels=224,
                                          width_pixels=224, scale=100, center=(112, 112), dim=2)
                    # Create an empty image
                    # Create an empty NumPy array
                    image_size = (224, 224, 3)
                    image = np.zeros(image_size, dtype=np.uint8)

                    # Set line colors as blue
                    num_nodes = len(x)
                    for i in range(num_nodes):
                        for j in range(i + 1, num_nodes):
                            if matrix[i, j] == 1:
                                x1, y1 = x[i], y[i]
                                x2, y2 = x[j], y[j]
                                #draw_line(image, x1, y1, x2, y2, (0, 0, 255))  # Draw line
                                if x1 == x2:
                                    for y_coord in range(min(y1, y2), max(y1, y2) + 1):
                                        image[x1, y_coord] = (0, 0, 255)  # Set pixel color (blue)
                                else:
                                    m = (y2 - y1) / (x2 - x1)
                                    for x_coord in range(min(x1, x2), max(x1, x2) + 1):
                                        y_coord = int(m * (x_coord - x1) + y1)
                                        image[x_coord, y_coord] = (0, 0, 255)  # Set pixel color (blue)

                    # Set node colors as red
                    for i in range(len(x)):
                        image[x[i], y[i]] = (255, 0, 0)  # Set pixel color (red)
                        image[x[i] - 1, y[i]] = (255, 0, 0)
                        image[x[i], y[i] - 1] = (255, 0, 0)
                        image[x[i] - 1, y[i] - 1] = (255, 0, 0)

                    os.makedirs(visualization + '_' + str(each) + '_' + model_size, exist_ok=True)
                    os.makedirs(visualization + '_' + str(each) + '_' + model_size + '_mat', exist_ok=True)

                    # Save the image

                    #image = image.resize((224, 224), resample=Image.NEAREST)
                    #image.save("output.png", "PNG", dpi=(300, 300), resample=Image.NEAREST)
                    imageio.imwrite(visualization + '_' + str(each) + '_' + model_size + '/' + str(each) + '_' + str(id_num) + '.png', image)
                    #image.save(
                    #    visualization + '_' + str(each) + '_' + model_size + '/' + str(each) + '_' + str(id_num) + '.png', dpi=(300, 300), resample=Image.NEAREST)
                    mat_file_name = visualization + '_' + str(each) + '_' + model_size + '_mat/' + str(each) + '_' + str(id_num)
                    np.save(mat_file_name, matrix)


print (id_num)