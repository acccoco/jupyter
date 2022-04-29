#! python3.9
############################################################
# 利用 fbm 生成有趣的纹理
############################################################
import time
import taichi as ti

ti.init(arch=ti.gpu)

# fbm iteration times
OCTAVES = 8

# rotate 36.87' clockwise
M_ROTATE = ti.Matrix([
    [0.8, -0.6],
    [0.6, 0.8]
])

resolution = 720, 720
color_buffer = ti.Vector.field(3, ti.f32, resolution)


@ti.func
def fract(x):
    return x - ti.floor(x)


@ti.func
def rand(st):
    return fract(ti.sin(ti.Vector([12.9898, 78.233]).dot(st)) * 43758.5453123)


@ti.func
def mix(a, b, x):
    return a * (1 - x) + b * x


@ti.func
def hermit(x):
    return x ** 2 * (3 - 2 * x)


@ti.func
def noise(p) -> float:
    p_i = ti.floor(p)

    # four corner value of a grid
    a = rand(p_i)
    b = rand(p_i + ti.Vector([1, 0]))
    c = rand(p_i + ti.Vector([0, 1]))
    d = rand(p_i + ti.Vector([1, 1]))

    uv_inter = hermit(p - p_i)
    return mix(
        mix(a, b, uv_inter[0]),
        mix(c, d, uv_inter[0]), uv_inter[1]
    )


@ti.func
def fbm(p) -> float:
    frequency = 1.0     # inital f
    amplitude = 0.5     # inital A
    data = 0.0

    for _ in range(OCTAVES):
        data += amplitude * noise(M_ROTATE @ p * frequency)

        # increase frequency, decrease amplitude
        amplitude *= 0.5
        frequency *= 2.0

    return data / (1.0 - 1.0/OCTAVES)


@ti.func
def pattern(p, itime: float) -> float:
    """
    p0 = p
    p1 = p + 4.0 * d1
    p2 = p + 4.0 * d2
    return: fbm(p2)
    """
    d1 = ti.Vector([
        fbm(p + ti.Vector([0.0, 0.0])),
        fbm(p + ti.Vector([5.2, 1.3]))
    ])
    d2 = ti.Vector([
        fbm(p + 4.0 * d1 + ti.Vector([1.7, 9.2])),
        fbm(p + 4.0 * d1 + ti.Vector([8.3, 2.8]))
    ])
    return fbm(p + 4.0 * d2 + itime)


@ti.kernel
def render(itime: float):
    for s, t in color_buffer:
        uv = ti.Vector([s, t]) / ti.Vector(resolution)

        # split to (5 x 5) tiles
        uv *= 5.0

        value = pattern(uv, itime)
        color_buffer[s, t] = ti.Vector([
            value * 0.8,
            value * 1.4,
            value * 2.4
        ])


gui = ti.GUI("wrapped fbm", resolution)
while gui.running:
    color_buffer.fill(ti.Vector([1.0, 1.0, 1.0]))
    render(time.time() % 1000)
    gui.set_image(color_buffer)
    gui.show()
