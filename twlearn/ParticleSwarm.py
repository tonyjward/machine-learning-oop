import math
import numpy as np

class Pso:

    def _initialise_swarm(self, upper, lower, no_features, no_particles):
        """ Initialise the velocity of the particles
        
        Arguments:
            upper -- scalar - upper limit on particle value
            lower -- scalar - lower limit on particle value
            no_features -- integer - number of features
            no_particles -- integer - number of particles in swarm

        Returns:
            position - position of the swarm - numpy array of size (no_features, no_particles)
        """
        position = np.random.uniform(low = lower, 
                                     high = upper, 
                                     size = (no_features, no_particles))
        return position

    def _update_velocity(self, velocity, historical_best, best_particle, swarm_position,
                               inertia, nostalgia, envy, no_features, no_particles):                                    
        """ Initialise the velocity of the particles
        
        Arguments:
            velocity            -- numpy matrix of size (no_features, no_particles)
            historical_best     -- numpy matrix of size (no_features, no_particles)
            best_particle       -- numpy matrix of size (no_features, 1)
            swarm_position      -- numpy matrix of size (no_features, no_particles)
            intertia --float    -- weight applied to velocity of particles when calculating new velocity,
            nostalgia -- float  -- weight applied to individual particles previous best when calculating velocity
            envy -- float       -- weight applied to swarm best when calculating velocity
            no_features         -- int, number of features
            no_particles        -- int, number of particles

        Returns:
            velocity - updated particle velocity numpy array of size (no_features, no_particles)
        """
        # check dimensions
        velocity_shape = velocity.shape

        # update velocity
        rand_weight_nostalgia = np.random.uniform(low = 0, 
                                                  high = 1, 
                                                  size = (no_features, no_particles))
        rand_weight_envy = np.random.uniform(low = 0, 
                                             high = 1, 
                                             size = (no_features, no_particles))  
        inertia_velocity = velocity * inertia
        nostalgia_velocity = (historical_best - swarm_position) * rand_weight_nostalgia * nostalgia
        envy_velocity = (best_particle - swarm_position) * rand_weight_envy * envy

        velocity = inertia_velocity + nostalgia_velocity + envy_velocity
        assert(velocity.shape == velocity_shape)

        return velocity

    def _predict_pso(self, X, particles):
        """ Make predictions on X matrix for supplied particles

        Arguments:
            X -- data matrix of size (no_examples, no_features_incl_bias)
            particles -- numpy matrix of size (no_features_incl_bias, no_particles)

        Returns:
            predictions -- numpy matrix of predictions of size (no_examples, no_particles)
        """
        assert(X.shape[1] == particles.shape[0])
        
        return np.dot(X, particles)


    def _update_historical_best(self, swarm_position, swarm_errors, 
                                 historical_best, historical_best_errors):
        """ Update historical best particle position and corresponding error

        Arguments:
            swarm_position          -- current position of each particle 
                                    -- numpy matrix of size (no_features, no_particles)
            swarm_errors            -- error of current particles positions 
                                    -- numpy matrix of size (1, no_particles)
            historical_best         -- historical best position of each particle
                                    -- numpy matrix of size (no_features, no_particles)
            historical_best_errors  -- errors of historic best position 
                                    -- numpy matrix of size (1, no_particles)

        Returns:
            historical_best         -- updated historical best 
            historical_best_errors  -- updated historical_best_errors

        Approach:
        If the current position of a particle in the swarm is better than the historical best
        then we update the historical_best and historical_best_errors matrices.
        Not sure whether this operation would be best done in the pso function, since 
        having a function requires us to overwrite the historical_best matrix each time (slow)
        """        
        # check dimensions
        historical_best_shape = historical_best.shape
        historical_best_errors_shape = historical_best_errors.shape

        # If any particles improve their position then update their history entry
        improved_particle_index = (swarm_errors < historical_best_errors).flatten()
        improved_particle_count = improved_particle_index.sum()

        if improved_particle_count > 0:
            historical_best[:, improved_particle_index] = swarm_position[:, improved_particle_index]
            historical_best_errors[:, improved_particle_index] = swarm_errors[:, improved_particle_index]
        
        assert(historical_best.shape == historical_best_shape)
        assert(historical_best_errors.shape == historical_best_errors_shape)

        return historical_best, historical_best_errors

    def _update_best_particle(self, historical_best, historical_best_errors):
        """ Update the best particle position and corresponding error
        Arguments:
            historical_best         -- best positions for each particle
                                    -- numpy matrix of size (no_features_incl_bias, no_particles)
            historical_best_errors  -- error of best position for each particle
                                    -- numpy matrix of size (1, no_particles)
        Returns:
            best_particle           -- best particle position found so far
                                    -- numpy matrix of size (no_features_incl_bias, 1)
            best_particle_error     -- corresponding error of best_particle
                                    -- scalar
        """
        best_particle_error = np.min(historical_best_errors)
        best_particle_index = np.argmin(historical_best_errors, axis = 1)
        best_particle = historical_best[:, best_particle_index]
        return best_particle, best_particle_error

    def _pso(self, X, y, num_iterations, no_particles, inertia, nostalgia, envy, upper, lower, loss):
        """ Perform particle swarm optimisaion

        Arguments:
 

        Returns:
            w -- weights - a numpy vector of size (no_features, 1)
            b -- bias - a scalar

        """
        # add bias 
        X = self._add_bias(X)
   
        no_examples, no_features_incl_bias = X.shape
        
        # Initialise swarm position
        swarm_position = self._initialise_swarm(upper = upper, 
                                                lower = lower, 
                                                no_features = no_features_incl_bias,
                                                no_particles = no_particles)
        assert(swarm_position.shape == (no_features_incl_bias, no_particles))
        
        swarm_predictions = self._predict_pso(X, swarm_position)
        assert(swarm_predictions.shape == (no_examples, no_particles))

        swarm_errors = loss(predictions = swarm_predictions, 
                            actual = y)
        assert(swarm_errors.shape == (1, no_particles))        
        
        # Intialise best historical position and errors
        historical_best = swarm_position.copy()
        historical_best_errors = swarm_errors.copy()
        
        # Find best particle and corresponding error
        best_particle, best_particle_error = self._update_best_particle(historical_best, 
                                                                        historical_best_errors)
        assert(best_particle.shape == (no_features_incl_bias, 1))

        # Initialise Particles Velocity
        velocity = np.random.uniform(low = -np.absolute(upper - lower), 
                                     high = np.absolute(upper - lower), 
                                     size = (no_features_incl_bias, no_particles))
        assert(velocity.shape == (no_features_incl_bias, no_particles))

        #--------WHILE A TERMINATION CRITERIAN IS NOT MET
        for i in range(num_iterations):

            #Update Velocity
            velocity = self._update_velocity(velocity = velocity, 
                                             historical_best = historical_best, 
                                             best_particle = best_particle, 
                                             swarm_position = swarm_position,
                                             inertia = inertia,  
                                             nostalgia = nostalgia,
                                             envy = envy, 
                                             no_features = no_features_incl_bias,
                                             no_particles = no_particles)
            assert(velocity.shape == (no_features_incl_bias, no_particles))

            # Update particles position
            swarm_position = swarm_position + velocity

            swarm_predictions = self._predict_pso(X, swarm_position)
            assert(swarm_predictions.shape == (no_examples, no_particles))

            swarm_errors = loss(predictions = swarm_predictions, 
                                actual = y)
            assert(swarm_errors.shape == (1, no_particles))   

            # Update historical best and historical error for each particle if new position better   
            historical_best, historical_best_errors = self._update_historical_best(swarm_position,
                                                                                   swarm_errors,
                                                                                   historical_best,
                                                                                   historical_best_errors)
            assert(historical_best.shape == (no_features_incl_bias, no_particles))

            best_particle, best_particle_error = self._update_best_particle(historical_best, 
                                                                            historical_best_errors)
        b = best_particle[0]
        w = best_particle[1:]
        return w, b
