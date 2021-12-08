#! python3.9
############################################################
# 二维的随机数
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
    return fract(ti.sin(ti.Vector([12.9898, 78.233]).dot(st)) * 43758.5453123)


@ti.func
def hermit(x):
    """
    Hermit interpolation, cubic
    """
    return x ** 2 * (3 - 2 * x)


@ti.func
def mix(a, b, x):
    return a * (1 - x) + b * x


@ti.func
def noise(uv):
    uv_i = ti.floor(uv)
    a = rand2(uv_i)
    b = rand2(uv_i + ti.Vector([1, 0]))
    c = rand2(uv_i + ti.Vector([0, 1]))
    d = rand2(uv_i + ti.Vector([1, 1]))

    uv_inter = hermit(uv - uv_i)
    return mix(
        mix(a, b, uv_inter[0]),
        mix(c, d, uv_inter[0]), uv_inter[1]
    )


@ti.kernel
def render(t: float):
    for x, y in color_buffer:
        st = ti.Vector([x, y]) / ti.Vector(resolution) * 5.0 + t
        color = noise(st)
        color_buffer[x, y] = ti.Vector([color, color, color])


if __name__ == "__main__":
    gui = ti.GUI("random", resolution)
    while gui.running:
        color_buffer.fill(ti.Vector([1.0, 1.0, 1.0]))
        render(time.time() % 100)
        gui.set_image(color_buffer)
        gui.show()
