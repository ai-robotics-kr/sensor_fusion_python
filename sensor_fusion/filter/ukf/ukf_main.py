class UnscentedKalmanFilter(object):
    def __init__(self, n_x, n_y, dt, fn_f, fn_h, fn_sigma, fn_utf, mean_init=None, cov_init=None):
        """Unscented Kalman Filter class.

        Args:
            n_x (int): Dimension of the state
            n_y (int): Dimension of the measurement
            dt (float): Timestep
            fn_f (function): Process function
            fn_h (function): Measurement function
            fn_sigma (function): Sigma point generating function
            mean_init (np.ndarray): Initial mean to sample the sigma points from
            cov_init (np.ndarray): Initial covariance to sample the sigma points from
        """
        self.n_x = n_x
        self.n_y = n_y
        self.dt = dt
        self.f = fn_f
        self.h = fn_h
        self.gen_sigma = fn_sigma
        self.utf = fn_utf
        self.Q = np.eye(n_x)
        self.R = np.eye(n_y)

        self.sigma = None
        self.w_mean = None
        self.w_cov = None

        self.sigma_prior = None
        self.x_prior = None
        self.P_prior = None

        self.x_posterior = mean_init
        self.P_posterior = cov_init
        if not isinstance(mean_init, np.ndarray):
            self.x_posterior = np.zeros(n_x)
        if not isinstance(cov_init, np.ndarray):
            self.P_posterior = np.eye(n_x)

        self.K = None # Kalman gain
    
    def predict(self):
        # generate sigma points
        alpha = .3
        beta = 2.
        kappa = .1
        (sigma, w_mean, w_cov) = generate_vandermerwe_sigma_point(self.x_posterior, self.P_posterior, self.n_x, alpha, beta, kappa)
        self.num_sigma = sigma.shape[0]
        self.sigma = sigma
        self.w_mean = w_mean
        self.w_cov = w_cov

        # propagate sigma points
        sigma_prop = np.zeros((self.num_sigma, self.n_x))
        for idx in range(self.num_sigma):
            sigma_prop[idx,:] = self.f(self.sigma[idx], self.dt)
        mean_unscented, cov_unscented = self.utf(sigma_prop, self.w_mean, self.w_cov, self.Q)

        self.sigma_prior = sigma_prop
        self.x_prior = mean_unscented
        self.P_prior = cov_unscented

    def update(self, measurement):
        # propagate the sigma points through the measurement function
        sigma_prop = np.zeros((self.num_sigma, self.n_y))
        for idx in range(self.num_sigma):
            sigma_prop[idx,:] = self.h(self.sigma_prior[idx])
        
        # compute the mean and covariance using the unscented transform
        mean_y, cov_y = self.utf(sigma_prop, self.w_mean, self.w_cov, self.R)

        # compute the residual
        residual = measurement - mean_y

        # compute the Kalman gain
        # cross covariance
        P_xy = np.zeros((self.sigma_prior.shape[1], sigma_prop.shape[1]))
        for idx in range(self.num_sigma):
            P_xy += self.w_cov[idx]*np.outer(self.sigma_prior[idx,None,:].T-self.x_prior, 
                                          sigma_prop[idx,None,:].T-mean_y)
        self.K = P_xy.dot(np.linalg.inv(cov_y))

        self.x_posterior = self.x_prior + self.K.dot(residual)
        self.P_posterior = self.P_prior - self.K.dot(cov_y).dot(self.K.T)



def Q_discrete_white_noise(dim, dt=1., var=1., block_size=1, order_by_dim=True):
    if dim not in [2, 3, 4]:
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [[.25*dt**4, .5*dt**3],
             [ .5*dt**3,    dt**2]]
    elif dim == 3:
        Q = [[.25*dt**4, .5*dt**3, .5*dt**2],
             [ .5*dt**3,    dt**2,       dt],
             [ .5*dt**2,       dt,        1]]
    else:
        Q = [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
             [(dt**5)/12, (dt**4)/4,  (dt**3)/2, (dt**2)/2],
             [(dt**4)/6,  (dt**3)/2,   dt**2,     dt],
             [(dt**3)/6,  (dt**2)/2 ,  dt,        1.]]
    return np.asarray(Q)*var


np.random.seed(1234)
std_x, std_y = .3, .3
ys = [np.array([[idx + np.random.randn()*std_x],
                [idx + np.random.randn()*std_y]]) for idx in range(100)]

def f_cv(x, dt):
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    return F@x

def h_cv(x):
    return x[[0,2]]

ukf = UnscentedKalmanFilter(n_x=4, n_y=2, dt=1.0, fn_f=f_cv, fn_h=h_cv,
                            fn_sigma=generate_vandermerwe_sigma_point, 
                            fn_utf=unscented_transform,
                            mean_init=np.zeros((1,4)))
ukf.R = np.diag([0.09, 0.09])
ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)

uxs = []
xs = []
for idx in range(len(ys)):
    ukf.predict()
    ukf.update(ys[idx])
    xs.append(ukf.x_posterior)

measurement = np.asarray(ys)[:,:,0]
estimated = np.asarray(xs)[:,:,0]

print("Done")


plt.plot(measurement[:,0], measurement[:,1], linewidth=3)
plt.plot(estimated[:,0], estimated[:,2], linewidth=3, alpha=0.5)
plt.show()


# ========== Airplane tracking
import math

def f_radar(x, dt):
    F = np.array([[1, dt, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=float)
    return F@x

def h_radar(x):
    dx = x[0] - h_radar.radar_pos[0]
    dy = x[2] - h_radar.radar_pos[1]
    slant_range = math.sqrt(dx**2 + dy**2)
    elevation_angle = math.atan2(dy,dx)
    return [slant_range, elevation_angle]

h_radar.radar_pos = (0,0)

class RadarStation:
    def __init__(self, pos, range_std, elev_angle_std):
        self.pos = np.asarray(pos)
        self.range_std = range_std
        self.elev_angle_std = elev_angle_std
    
    def reading_of(self, ac_pos):
        """Returns (range, elevation angle) to aircraft.
        Elevation angle is in radians.
        """
        diff = np.subtract(ac_pos, self.pos)
        rng = np.linalg.norm(diff)
        brg = math.atan2(diff[1], diff[0])
        return rng, brg

    def noisy_reading(self, ac_pos):
        """Compute range and elevation angle to aircraft with simulated noise.
        """
        rng, brg = self.reading_of(ac_pos)
        rng += np.random.randn() * self.range_std
        brg += np.random.randn() * self.elev_angle_std
        return rng, brg

class ACSim:
    def __init__(self, pos, vel, vel_std):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.vel_std = vel_std
    
    def update(self, dt):
        """Compute and return next position.
        Incorporates random variation in velocity."""
        dx = self.vel*dt + (np.random.randn() * self.vel_std) * dt
        self.pos += dx
        return self.pos

# ========== Track an airplane
dt = 3.
range_std = 5
elevation_angle_std = math.radians(0.5)
ac_pos = (0., 1000.)
ac_vel = (100., 0.)
radar_pos = (0., 0.)
h_radar.radar_pos = radar_pos

ukf = UnscentedKalmanFilter(n_x=3, n_y=2, dt=dt, 
                            fn_f=f_radar, 
                            fn_h=h_radar, 
                            fn_sigma=generate_vandermerwe_sigma_point, 
                            fn_utf=unscented_transform)
ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
ukf.Q[2,2] = 0.1
ukf.R = np.diag([range_std**2, elevation_angle_std**2])
ukf.x = np.array([0., 90., 1100.])
ukf.P = np.diag([300**2, 30**2, 150**2])

np.random.seed(200)
pos = (0,0)
radar = RadarStation(pos, range_std, elevation_angle_std)
ac = ACSim(ac_pos, ac_vel, 0.02)

time = np.arange(0, 360+dt, dt)
xs = []
for _ in time:
    ac.update(dt)
    r = radar.noisy_reading(ac.pos)
    ukf.predict()
    ukf.update(np.array([[r[0], r[1]]]).T)
    xs.append(ukf.x_posterior)

estimated = np.asarray(xs)[:,:,0]

fig, axes = plt.subplots(1,3)
axes[0].set_title("Position")
axes[1].set_title("Velocity")
axes[2].set_title("Altitude")
axes[0].plot(estimated[:,0], linewidth=3)
axes[1].plot(estimated[:,1], linewidth=3)
axes[2].plot(estimated[:,2], linewidth=3)
plt.show()


# ========== Track climbing airplane w/o modification
ukf.x_posterior = np.array([0., 90., 1100.])
ukf.P_posterior = np.diag([300**2, 30**2, 150**2])
ac = ACSim(ac_pos, ac_vel, 0.02)

np.random.seed(200)
time = np.arange(0, 360+dt, dt)
xs, ys = [], []
for t in time:
    if t >= 60:
        ac.vel[1] = 300/60
    ac.update(dt)
    r = radar.noisy_reading(ac.pos)
    ys.append(ac.pos[1])
    ukf.predict()
    ukf.update(np.array([[r[0], r[1]]]).T)
    xs.append(ukf.x_posterior)

measurement = np.asarray(ys)
estimated = np.asarray(xs)[:,:,0]
fig, axes = plt.subplots(1,3)
axes[0].set_title("Position")
axes[1].set_title("Velocity")
axes[2].set_title("Altitude")
axes[0].plot(estimated[:,0], linewidth=3)
axes[1].plot(estimated[:,1], linewidth=3)
axes[2].plot(estimated[:,2], linewidth=3)
plt.show()

plt.plot(measurement, linewidth=3)
plt.plot(estimated[:,2], linewidth=3, alpha=0.5)
plt.show()


# ========== Track climbing airplane with modification
def f_cv_radar(x, dt):
    F = np.array([[1, dt, 0, 0],
                  [0,  1, 0, 0],
                  [0,  0, 1, dt],
                  [0,  0, 0, 1]], dtype=float)
    return F @ x


    
ukf = UnscentedKalmanFilter(n_x=4, n_y=2, dt=dt, 
                            fn_f=f_cv_radar, 
                            fn_h=h_radar, 
                            fn_sigma=generate_vandermerwe_sigma_point, 
                            fn_utf=unscented_transform)
ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.1)
ukf.R = np.diag([range_std, elevation_angle_std]) @ np.diag([range_std, elevation_angle_std])
ukf.x = np.array([0., 90., 1100., 0.])
ukf.P = np.diag([300**2, 3**2, 150**2, 3**2])

pos = (0,0)
radar = RadarStation(pos, range_std, elevation_angle_std)
ac = ACSim(ac_pos, ac_vel, 0.02)

np.random.seed(200)
time = np.arange(0, 360+dt, dt)
xs, ys = [], []
for t in time:
    if t >= 60:
        ac.vel[1] = 300/60
    ac.update(dt)
    r = radar.noisy_reading(ac.pos)
    ys.append(ac.pos[1])
    ukf.predict()
    ukf.update(np.array([[r[0], r[1]]]).T)
    xs.append(ukf.x_posterior)

measurement = np.asarray(ys)
estimated = np.asarray(xs)[:,:,0]
fig, axes = plt.subplots(1,3)
axes[0].set_title("Position")
axes[1].set_title("Velocity")
axes[2].set_title("Altitude")
axes[0].plot(time, estimated[:,0], linewidth=3)
axes[1].plot(time, estimated[:,1], linewidth=3)
axes[2].plot(time, estimated[:,2], linewidth=3)
plt.show()

plt.plot(time, measurement, linewidth=3)
plt.plot(time, estimated[:,2], linewidth=3, alpha=0.5)
plt.show()


# ========== Sensor fusion
# -------- (w/o fusion)
range_std = 500.
elevation_angle_std = math.radians(0.5)
np.random.seed(200)
pos = (0., 0.)
radar = RadarStation(pos, range_std, elevation_angle_std)
ac = ACSim(ac_pos, (100, 0), 0.02)

ukf = UnscentedKalmanFilter(n_x=4, n_y=2, dt=dt, 
                            fn_f=f_cv_radar, 
                            fn_h=h_radar, 
                            fn_sigma=generate_vandermerwe_sigma_point, 
                            fn_utf=unscented_transform)
ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.1)
ukf.R = np.diag([range_std, elevation_angle_std]) @ np.diag([range_std, elevation_angle_std])
ukf.x = np.array([0., 90., 1100., 0.])
ukf.P = np.diag([300**2, 3**2, 150**2, 3**2])

time = np.arange(0, 360+dt, dt)
xs, ys = [], []
for t in time:
    # if t >= 60:
    #     ac.vel[1] = 300/60
    ac.update(dt)
    r = radar.noisy_reading(ac.pos)
    ys.append(ac.pos[1])
    ukf.predict()
    ukf.update(np.array([[r[0], r[1]]]).T)
    xs.append(ukf.x_posterior)

measurement = np.asarray(ys)
estimated = np.asarray(xs)[:,:,0]

plt.plot(time, estimated[:,1], linewidth=3)
plt.show()

print("Velocity tracking performance has definitely degraded!")
print("Standard deviation: " + str(np.std(estimated[10:,1])))

# -------- (w/ fusion)
def h_vel(x):
    dx = x[0] - h_vel.radar_pos[0]
    dz = x[2] - h_vel.radar_pos[1]
    slant_range = math.sqrt(dx**2 + dz**2)
    elevation_angle = math.atan2(dz, dx)
    return slant_range, elevation_angle, x[1], x[3]

h_radar.radar_pos = (0, 0)
h_vel.radar_pos = (0, 0)

range_stsd = 500.
elevation_angle_std = math.radians(0.5)
vel_std = 2.

np.random.seed(200)
ac = ACSim(ac_pos, (100, 0), 0.02)
radar = RadarStation((0, 0), range_std, elevation_angle_std)

ukf = UnscentedKalmanFilter(n_x=4, n_y=4, dt=dt, 
                            fn_f=f_cv_radar, 
                            fn_h=h_vel, 
                            fn_sigma=generate_vandermerwe_sigma_point, 
                            fn_utf=unscented_transform)
ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.1)
ukf.R = np.diag([range_std, elevation_angle_std, vel_std, vel_std]) @ \
        np.diag([range_std, elevation_angle_std, vel_std, vel_std])
ukf.x = np.array([0., 90., 1100., 0.])
ukf.P = np.diag([300**2, 3**2, 150**2, 3**2])

time = np.arange(0, 360+dt, dt)
xs, ys = [], []
for t in time:
    # if t >= 60:
    #     ac.vel[1] = 300/60
    ac.update(dt)
    r = radar.noisy_reading(ac.pos)
    vx = ac.vel[0] + np.random.randn()*vel_std
    vz = ac.vel[1] + np.random.randn()*vel_std
    ukf.predict()
    ukf.update(np.array([[r[0], r[1], vx, vz]]).T)
    xs.append(ukf.x_posterior)

measurement = np.asarray(ys)
estimated = np.asarray(xs)[:,:,0]

plt.plot(time, estimated[:,1], linewidth=3)
plt.show()

print("Standard deviation: " + str(np.std(estimated[10:,1])))

