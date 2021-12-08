#! python3
############################################################
# 一些重复的模式
############################################################

import math
import taichi as ti

ti.init(arch=ti.gpu)

res = 360, 360
buffer = ti.Vector.field(3, ti.f32, res)


@ti.func
def rotate2d(st, angle):
    m = ti.Matrix([
        [ti.cos(angle), ti.sin(angle)],
        [-ti.sin(angle), ti.cos(angle)]
    ])

    return m @ (st - 0.5) + 0.5


@ti.kernel
def render():
    for x, y in buffer:
        st = ti.Vector([x, y]) / ti.Vector(res) * 3.0
        st = st - ti.floor(st)
        st = rotate2d(st, math.pi * 0.0)
        buffer[x, y] = ti.Vector([st[0], st[1], 0])


if __name__ == "__main__":
    gui = ti.GUI("random", res)
    while gui.running:
        buffer.fill(ti.Vector([1.0, 1.0, 1.0]))
        render()
        gui.set_image(buffer)
        gui.show()
