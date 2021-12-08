#! python3.9
############################################################
# cellular noise
############################################################
import time
import taichi as ti

ti.init(arch=ti.gpu)

resolution = 720, 720
color_buffer = ti.Vector.field(3, ti.f32, resolution)


@ti.func
def fract(x):
    return x - ti.floor(x)


@ti.func
def rand2(st):
    return fract(ti.sin(ti.Vector([
        ti.Vector([127.1, 311.7]).dot(st),
        ti.Vector([269.5, 183.3]).dot(st)
    ])) * 43758.5453)


@ti.func
def distance(p1, p2):
    return (p1 - p2).norm()


@ti.kernel
def render(t: float):
    for x, y in color_buffer:
        # tile the plane
        st = ti.Vector([x, y]) / ti.Vector(resolution) * 3.0

        st_i = ti.floor(st)         # grid positin
        pix_pos = st - st_i        # pixel position in grid

        dis_min = 1.0
        # [-1, 0, 1] * [-1, 0, 1] stand for neighbor 9 grids
        for i in range(-1, 2):
            for j in range(-1, 2):

                # generate point in neighbor grid
                delta_neighbor = ti.Vector([i, j])
                pos_in_neighbor = rand2(st_i + delta_neighbor)
                pos_in_neighbor = 0.5 + 0.5 * \
                    ti.sin(t + 6.2841 * pos_in_neighbor)    # move point by time

                # calculate distance field
                dis_min = min(dis_min, distance(pix_pos,
                                                pos_in_neighbor + delta_neighbor))
        color = ti.Vector([1, 1, 1]) * dis_min

        # draw cell center
        if dis_min < 0.01:
            color = ti.Vector([0, 1, 0])

        # draw isolines
        if pix_pos[0] > 0.99 or pix_pos[1] > 0.98:
            color = ti.Vector([1, 0, 0])

        color_buffer[x, y] = color


if __name__ == "__main__":
    gui = ti.GUI("random", resolution)
    while gui.running:
        color_buffer.fill(ti.Vector([1.0, 1.0, 1.0]))
        render(time.time() % 100)
        gui.set_image(color_buffer)
        gui.show()
