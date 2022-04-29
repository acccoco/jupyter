import numpy as np
import matplotlib.pyplot as plt


def vec(theta, phi):
    """根据方位角生成单位向量"""
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.array([x, y, z])


def normalize(v):
    """
    将向量归一化
    """
    l = np.maximum(0.0001, np.sqrt(np.sum(v * v)))
    return v / l


def GGX_NDF(angle_m_n, alpha):
    """
    angle_m_n 表示微表面法线和宏观法线的夹角
    """
    angle_m_n = np.deg2rad(angle_m_n)
    cos_theta = np.maximum(0, np.cos(angle_m_n))
    alpha2 = alpha ** 2
    return alpha2 / np.pi / np.power(cos_theta ** 2 * (alpha2 - 1) + 1, 2)


def GGX_G(angle, alpha):
    """
    angle 表示夹角，一般是观测方向 v 与 n 的夹角或者光线方向 l 与 n 的夹角
    """
    k = alpha / 2
    cos_theta = np.maximum(0, np.cos(np.deg2rad(angle)))
    return cos_theta / (cos_theta * (1 - k) + k)


def Smith_G(angle_l_n, angle_v_n, alpha):
    """
    angle_l_n 表示光线方向 l 与宏观法线 n 的夹角
    """
    return GGX_G(angle_l_n, alpha) * GGX_G(angle_v_n, alpha)


def Fresnel_Schlick(F0, angle_v_h):
    """
    接收基础反射系数和入射角度值
    """
    theta = np.deg2rad(angle_v_h)
    cos_theta = np.maximum(0, np.cos(theta))
    return F0 + np.power((1 - F0) * (1 - cos_theta), 5.0)


v_vec = np.vectorize(vec, otypes=[np.ndarray])
v_normalize = np.vectorize(normalize, otypes=[np.ndarray])
v_GGX_NDF = np.vectorize(GGX_NDF, otypes=[float])
v_Smith_G = np.vectorize(Smith_G, otypes=[float])
v_Fresnel_Schlick = np.vectorize(Fresnel_Schlick, otypes=[float])


def draw(theta_v, phi_v, alpha):
    """
    观察方向的方位角作为参数
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # 球坐标系上的方位角
    v = vec(theta_v, phi_v)

    # 计算观察向量 v 和另一个向量的半程向量
    v_half_to_view = np.vectorize(
        lambda _: normalize((v + _) / 2.0), otypes=[np.ndarray])
    # 计算向量和法线 n(0, 0, 1) 的夹角
    v_angle_to_normal = np.vectorize(
        lambda _: np.rad2deg(np.arccos(_[2])), otypes=[float])
    # 计算向量和观察方向 v 之间的夹角
    v_angle_to_view = np.vectorize(
        lambda _: np.rad2deg(np.arccos(v.T @ _)), otypes=[float])   # 这里的运算表示向量点积

    # 光线方向的方位角
    angle_l_n, phi_l = np.meshgrid(
        np.linspace(0, 360, 90), np.linspace(0, 90, 60))
    # 用直角坐标系表示光线的方向
    x = np.sin(np.deg2rad(phi_l)) * np.cos(np.deg2rad(angle_l_n))
    y = np.sin(np.deg2rad(phi_l)) * np.sin(np.deg2rad(angle_l_n))
    z = np.cos(np.deg2rad(phi_l))

    l = v_vec(angle_l_n, phi_l)                 # 光线方向的向量
    h = v_half_to_view(l)                       # 观察方向 v 和光线 l 的半程向量
    angle_h_n = v_angle_to_normal(h)            # h 和 n 之间的夹角
    angle_l_n = v_angle_to_normal(l)            # l 和 n 之间的夹角
    angle_v_h = v_angle_to_view(h)              # v 和 h 之间的夹角
    angle_v_n = np.rad2deg(np.arccos(v[2]))     # v 和 v 之间的夹角

    D = v_GGX_NDF(angle_h_n, alpha)                     # 法线分布的密度
    F = v_Fresnel_Schlick(F0=0.5, angle_v_h=angle_v_h)      # Fresnel 项
    G = v_Smith_G(angle_l_n, angle_v_n, alpha)          # Geometry 项

    len_mod = np.sqrt(D * F * G)        # 光线方向的 l 向量提供方程，这个变量提供长度
    ax.plot_surface(x * len_mod, y * len_mod, z *
                    len_mod, cmap=plt.cm.YlGnBu_r)

    plt.show()


if __name__ == "__main__":
    # 绘制 BRDF 的反射波瓣，前两个是观察方向的方位角度（球坐标系），第三个参数是粗糙度相关的 alpha
    draw(180, 45, 0.2)
