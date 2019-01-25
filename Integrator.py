class LeapfrogIntegrator:
    """Integrator that uses leapfrog (velocity - position offset) integration"""
    def __init__(self):
        pass

    def integrate(self, x_t, v_t_minus_half, f_t, m, dt):
        """
        Integrates using leapfrog method
        :param x_t: position at time t
        :param v_t_minus_half: velocity at time t - half time step
        :param f_t: force at time t
        :param m: mass
        :param dt: timestep
        :return: returns position at time t+1 and velocity at time t + half timestep
        """
        v_t_plus_half = v_t_minus_half + f_t/m*dt
        x_t_plus_one = x_t + v_t_plus_half*dt

        return x_t_plus_one, v_t_plus_half

    def push_back_velocity_half_step(self, v_t, f_t, m, dt):
        """
        Leapfrog requires that the initial velocity is pushed back to half a time step (to offset). This function
        does that.
        :param v_t: initial velocity
        :param f_t: intiial force
        :param m: mass
        :param dt: timestep
        :return: returns velocity at half a timestep before the initial velocity
        """
        # push back velocity half a time step
        v_t_minus_half = v_t - f_t/m*dt/2

        return v_t_minus_half