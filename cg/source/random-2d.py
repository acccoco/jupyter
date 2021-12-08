#! python3
############################################################
# 二维的随机数
############################################################
import taichi as ti

ti.init(arch=ti.gpu)

resolution = 720, 720
color_buffer = ti.Vector.field(3, ti.f32, resolution)


@ti.func
def fract(x):
    return x - ti.floor(x)


@ti.func
def rand2(st):
    return fract(ti.sin(ti.Vector([12.9898, 78.233]).dot(st)) * 43758.5453123)


@ti.kernel
def render():
    for x, y in color_buffer:
        color = rand2(ti.Vector([x, y]))

        color_buffer[x, y] = ti.Vector([color, color, color])


if __name__ == "__main__":
    gui = ti.GUI("random", resolution)
    while gui.running:
        color_buffer.fill(ti.Vector([1.0, 1.0, 1.0]))
        render()
        gui.set_image(color_buffer)
        gui.show()
