import numpy as np

from nptdsl import Grid1D, Field1D, d_t, d_xx, Equation, TimeStepper


def test_grid1d_basic():
    g = Grid1D(nx=11, length=2.0)
    assert g.nx == 11
    assert g.length == 2.0

    x = g.x
    assert x.shape == (11,)
    assert np.isclose(x[0], 0.0)
    assert np.isclose(x[-1], 2.0)

    # dx * (nx-1) 应该等于 length
    assert np.isclose(g.dx * (g.nx - 1), g.length)


def test_field1d_dirichlet():
    g = Grid1D(nx=5, length=1.0)
    u = Field1D("u", g)

    u.values[:] = 1.0
    u.set_dirichlet(left=0.0, right=2.0)

    assert np.isclose(u.values[0], 0.0)
    assert np.isclose(u.values[-1], 2.0)
    # 中间不变
    assert np.allclose(u.values[1:-1], 1.0)


def _explicit_heat_step(values, alpha, dx, dt):
    """和 TimeStepper 一致的 1D 显式 Euler 步进，用来做对照。"""
    old = values.copy()
    new = old.copy()
    coeff = alpha / (dx * dx)
    new[1:-1] = (
        old[1:-1]
        + dt * coeff * (old[0:-2] - 2.0 * old[1:-1] + old[2:])
    )
    # 边界保持不变（Dirichlet 由外部管理）
    return new


def test_time_stepper_matches_explicit_scheme():
    # 一维区域 [0, 1]，sin(pi x) 初值，0-Dirichlet 边界
    nx = 101
    length = 1.0
    alpha = 0.1
    dt = 1e-4

    g = Grid1D(nx=nx, length=length)
    u = Field1D("u", g)

    x = g.x
    u.values[:] = np.sin(np.pi * x)
    u.set_dirichlet(left=0.0, right=0.0)

    # 建方程 dt(u) = alpha * dxx(u)
    eq = Equation(d_t(u), alpha * d_xx(u))
    stepper = TimeStepper(eq).build()

    # 保存一份 old，用来自己手算一步
    old_vals = u.values.copy()
    # TimeStepper 做一步
    stepper.step(u, dt)

    # 手算一步
    expected = _explicit_heat_step(old_vals, alpha, g.dx, dt)

    assert np.allclose(u.values, expected, rtol=1e-7, atol=1e-9)


def test_time_stepper_smooths_peak():
    # 中间一个尖峰，0 Dirichlet 边界，热方程应当把尖峰抹平
    nx = 51
    g = Grid1D(nx=nx, length=1.0)
    u = Field1D("u", g)

    u.values[:] = 0.0
    center = nx // 2
    u.values[center] = 1.0
    u.set_dirichlet(left=0.0, right=0.0)

    alpha = 0.5
    dt = 1e-4
    t_final = 1e-2

    eq = Equation(d_t(u), alpha * d_xx(u))
    stepper = TimeStepper(eq).build()

    initial_center_value = u.values[center]
    stepper.run(u, t_final=t_final, dt=dt)
    final_center_value = u.values[center]

    # 中心的高度应该降低（扩散）
    assert final_center_value < initial_center_value
    # 所有值仍然非负（对这个简单配置而言）
    assert np.all(u.values >= -1e-12)
