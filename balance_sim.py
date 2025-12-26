import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Parameters
'''
    In the real world:
    - motors aren’t identical
    - COM isn’t perfectly centered
    - sensors have bias
    - friction isn’t symmetric

    So a robot often balances at:
    - 0.1°
    - 0.3°
    - 1°
    *These are excellent values*
'''

#Physics + control parameters
g = 9.81
L = 0.20
ell = L/2

# PD Controller -> stabilize quickly, damp motion, does not guarantee zero final error

# IMU realism (noise + bias drift)
angle_bias = 0.0                 # degrees (accel tilt bias)
bias_drift_per_step = 0.00002    # degrees/step (slow drift)
angle_noise_std = 0.05           # degrees (accel noise)
gyro_noise_std = 0.2             # deg/s (gyro noise)

alpha = 0.98   # complementary filter: trust gyro 98%, accel 2%

dt = 0.01  # time step (s)

# ---- Cart-pole state ----
x = 0.0          # cart position (m)
x_dot = 0.0      # cart velocity (m/s)

angle = 10.0     # pole angle (degrees), 0 = upright
# angle > 0  -> pole leaning RIGHT
# angle < 0  -> pole leaning LEFT

angular_velocity = 0.0  # deg/s
integral = 0.0          # (optional) integral on angle, keep mostly 0 for now

# Initialize fused angle ONCE (don’t reset each loop)
fused_angle = angle   # start with some error

def wrap_deg(a):
    # wrap to [-180, 180)
    return (a + 180.0) % 360.0 - 180.0

def angle_error_upright(angle_deg):
    # error relative to upright (0 deg), wrapped
    return wrap_deg(angle_deg)


def hybrid_controller(fused_angle_deg, measured_gyro_dps, x, x_dot, integral):
    """
    Two modes:
      - swing-up when far from upright
      - balance (PD) when near upright
    """

    # --- cart centering (always on, prevents runaway) ---
    Kp_x = 1.0
    Kd_x = 1.8
    F_center = -(Kp_x * x + Kd_x * x_dot)

    # Angle error around upright
    err_deg = angle_error_upright(fused_angle_deg)
    theta = math.radians(err_deg)
    theta_dot = math.radians(measured_gyro_dps)

    # --- decide mode ---
    # If within this window, do balancing
    balance_window_deg = 30.0

    if abs(err_deg) < balance_window_deg:
        # ---- BALANCE MODE (PD) ----
        Kp_theta = 55.0
        Kd_theta = 10.0
        Ki_theta = 0.0

        integral += theta * dt

        deadband = math.radians(0.15)
        if abs(theta) < deadband:
            theta = 0.0

        F_bal = -(Kp_theta * theta + Kd_theta * theta_dot + Ki_theta * integral)
        F = F_bal + F_center

    else:
        # ---- SWING-UP MODE (energy shaping) ----
        # Treat the pole like a pendulum and pump its energy until it reaches upright.

        # Normalize angle to radians around upright (0 rad).
        # Upright is 0, hanging down is ±pi.
        theta_u = theta  # theta is already error-to-upright in radians from your earlier code
        theta_dot_u = theta_dot

        # Energy of pendulum relative to hanging-down equilibrium:
        # E = 0 at down, increases as you swing up.
        # For a point mass at ell: E = 0.5*ell^2*theta_dot^2 + g*ell*(1 - cos(theta_down))
        # Using upright-error angle theta_u:
        # down corresponds to theta_u ≈ ±pi
        # Equivalent energy form: E = 0.5*ell^2*theta_dot^2 + g*ell*(1 + cos(theta_u))
        E = 0.5 * (ell**2) * (theta_dot_u**2) + g * ell * (1 + math.cos(theta_u))

        # Target energy for upright (theta_u=0, theta_dot=0):
        # E_target = g*ell*(1 + cos(0)) = 2*g*ell
        E_target = 2 * g * ell

        # Energy error
        E_err = E - E_target

        # Pumping direction:
        # multiply by sign(theta_dot * cos(theta)) is a common choice
        direction = 1.0 if (theta_dot_u * math.cos(theta_u)) >= 0 else -1.0

        # Swing gain (tune)
        K_E = 25.0

        F_swing = -K_E * E_err * direction

        # Add centering so cart doesn't run away
        F = F_swing + F_center
  

    # Clamp force (motor saturation)
    F_max = 60.0
    F = max(-F_max, min(F, F_max))

    return F, integral

def update_physics_cartpole(x, x_dot, angle_deg, angular_velocity_dps, force):
    """
    Standard cart-pole dynamics.

    Convention:
      angle_deg = 0  -> pole upright
      angle_deg = 180 or -180 -> pole hanging down
    """
    g = 9.81

    # masses + geometry
    M = 0.7      # cart mass (kg)
    m = 0.3      # pole mass (kg)
    L = 0.20     # pole length pivot->end (m)
    ell = L / 2  # center of mass distance (m)

    # damping/friction
    b_cart = 0.10     # N*s/m
    b_pole = 0.02     # N*m*s/rad

    # Convert to radians
    theta = math.radians(angle_deg)
    theta_dot = math.radians(angular_velocity_dps)

    # Wrap theta to [-pi, pi] to avoid it drifting to huge angles
    # (helps display + numerical stability)
    theta = (theta + math.pi) % (2 * math.pi) - math.pi

    # Apply cart friction
    F = force - b_cart * x_dot

    total_mass = M + m
    polemass_length = m * ell

    s = math.sin(theta)
    c = math.cos(theta)

    # This is the standard structure:
    temp = (F + polemass_length * theta_dot**2 * s) / total_mass

    # Add pole damping as torque opposing theta_dot
    # Convert damping torque into an equivalent angular acceleration term
    # tau_d = -b_pole*theta_dot, I_eff ≈ m*ell^2
    I_eff = max(m * ell**2, 1e-6)
    theta_damp_acc = (-b_pole * theta_dot) / I_eff

    # theta_ddot and x_ddot (standard cartpole)
    theta_ddot = (g * s - c * temp) / (ell * (4.0/3.0 - (m * c * c) / total_mass))
    theta_ddot += theta_damp_acc

    x_ddot = temp - (polemass_length * theta_ddot * c) / total_mass

    # Integrate (semi-implicit Euler)
    x_dot += x_ddot * dt
    x += x_dot * dt

    theta_dot += theta_ddot * dt
    theta += theta_dot * dt

    # Convert back to degrees
    angle_deg = math.degrees(theta)
    angular_velocity_dps = math.degrees(theta_dot)

    return x, x_dot, angle_deg, angular_velocity_dps

# ---------------- ANIMATION Controls ----------------

# Drawing sizes (purely visual)
L_draw = 1.0                  # rod length on screen
cart_w, cart_h = 0.7, 0.25     # cart size on screen
wheel_r = 0.08
DRAW_SCALE = 2.0              # meters -> screen units (visual scaling)

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(-3.0, 3.0)
ax.set_ylim(-0.2, 2.0)
ax.set_title("Cart + Inverted Pendulum (IMU + Complementary Filter + Control)")

# Ground line
ax.plot([-10, 10], [0, 0])

# Cart body (rectangle)
cart = Rectangle((0.0 - cart_w/2, 0.0), cart_w, cart_h, fill=False, linewidth=2)
ax.add_patch(cart)

# Wheels
wheel1, = ax.plot([], [], marker="o", markersize=12)
wheel2, = ax.plot([], [], marker="o", markersize=12)

# Rod + bob
rod_line, = ax.plot([], [], linewidth=3)
bob_point, = ax.plot([], [], marker="o", markersize=12)

# Text overlay
info = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")


def sim_step():
    """Run ONE time-step of simulation and return values for display."""
    global x, x_dot, angle, angular_velocity, integral, angle_bias, fused_angle

    # --- sensor readings (accel + gyro) ---
    angle_bias += random.uniform(-bias_drift_per_step, bias_drift_per_step)

    measured_angle = angle + angle_bias + random.gauss(0.0, angle_noise_std)          # accel-like
    measured_gyro = angular_velocity + random.gauss(0.0, gyro_noise_std)              # gyro-like

    # --- complementary filter (MPU6050-style) ---
    gyro_angle_est = fused_angle + measured_gyro * dt
    fused_angle = alpha * gyro_angle_est + (1 - alpha) * measured_angle

    # --- controller outputs FORCE on cart ---
    force, integral = force, integral = hybrid_controller(fused_angle, measured_gyro, x, x_dot, integral)


    # --- physics update (cart moves + pole rotates) ---
    x, x_dot, angle, angular_velocity = update_physics_cartpole(x, x_dot, angle, angular_velocity, force)

    # Errors (for insight)
    meas_err = measured_angle - angle
    fused_err = fused_angle - angle

    return measured_angle, fused_angle, force, meas_err, fused_err


def init():
    rod_line.set_data([], [])
    bob_point.set_data([], [])
    wheel1.set_data([], [])
    wheel2.set_data([], [])
    info.set_text("")
    return rod_line, bob_point, wheel1, wheel2, cart, info


def animate(frame):
    measured_angle, fused, F, meas_err, fused_err = sim_step()

    # Draw cart using x (scaled for screen)
    cart_x = x * DRAW_SCALE

    # Keep cart on screen (visual only; doesn’t change physics)
    cart_x = max(-2.5, min(2.5, cart_x))

    cart.set_xy((cart_x - cart_w/2, 0.0))

    # Wheels positions
    w1x = cart_x - cart_w * 0.25
    w2x = cart_x + cart_w * 0.25
    wy = wheel_r
    wheel1.set_data([w1x], [wy])
    wheel2.set_data([w2x], [wy])

    # Pivot point at top center of cart
    pivot_x = cart_x
    pivot_y = cart_h

    # Draw rod using TRUE angle (physics truth)
    theta = math.radians(angle)
    bob_x = pivot_x + L_draw * math.sin(theta)
    bob_y = pivot_y + L_draw * math.cos(theta)

    rod_line.set_data([pivot_x, bob_x], [pivot_y, bob_y])
    bob_point.set_data([bob_x], [bob_y])

    info.set_text(
        f"step={frame}\n"
        f"x={x:+.3f} m, x_dot={x_dot:+.3f} m/s\n"
        f"true={angle:+.3f}°\n"
        f"meas={measured_angle:+.3f}°  (err={meas_err:+.3f}°)\n"
        f"fused={fused:+.3f}° (err={fused_err:+.3f}°)\n"
        f"F={F:+.3f} N"
    )

    return rod_line, bob_point, wheel1, wheel2, cart, info


ani = FuncAnimation(fig, animate, init_func=init, frames=2000, interval=10, blit=True)
plt.show()
