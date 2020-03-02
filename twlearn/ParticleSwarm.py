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

    def _update_velocity(self, velocity, historical_best, swarm_best, swarm_position,
                               inertia, nostalgia, envy, no_features, no_particles):                                    
        """ Initialise the velocity of the particles
        
        Arguments:
            velocity --
            historical_best --
            swarm_best --
            swarm_position --
            intertia --,
            nostalgia --
            envy --
            no_features --
            no_particles --

        Returns:
            velocity - updated particle velocity numpy array of size (no_features, no_particles)
        """
        rand_weight_nostalgia = np.random.uniform(low = 0, 
                                                  high = 1, 
                                                  size = (no_features, no_particles))
        rand_weight_envy = np.random.uniform(low = 0, 
                                             high = 1, 
                                             size = (no_features, no_particles))  
        inertia_velocity = velocity * inertia
        nostalgia_velocity = (historical_best - swarm_position) * rand_weight_nostalgia * nostalgia
        envy_velocity = (swarm_best - swarm_position) * rand_weight_envy * envy
        velocity = innertia_velocity + nostalgia_velocity + envy_velocity
        
        return velocity

    def _predict_pso(self, X, particles):
        """ Make predictions on X matrix for supplied particles

        Arguments:
            X -- data matrix of size (no_examples, no_features_incl_bias)
            particles -- numpy matrix of size (no_features_incl_bias, no_particles)

        Returns:
            predictions -- numpy matrix of predictions of size (no_examples, no_features_incl_bias)
        """
        assert(X.shape[1] == particles.shape[0])
        
        return np.dot(X, particles)

    def _mae_pso(self, predictions, actual):
        """
        Calculate Mean Absolute Error

        Arguments:
            predictions: predictions numpy array of size (no_examples, no_particles)
            actuals: 1D numpy array of size (no_examples, 1)

        Returns:
            mae: mae for each particle - numpy array of size (1, no_particles)
        """
        assert(predictions.shape[0] == actual.shape[0])
        absolute_errors = np.abs(predicted - actual)
        return np.mean(absolute_errors, axis = 0)

    def _update_swarm_best(self, swarm_position, swarm_errors, 
                                 historical_best, historical_best_errors):
        
        # compare min(swarm_errors) to min(historical_best_errors)
        # if min(historical_best_errors) is lowest then return corresponding column from historical_best
        # if min(swarm_errors) is lowest then return corresponding column from swarm_position

        # # find particle with minimum error
        # minimum_error = np.amin(errors_champion)

        # # TODO: check uniqueness of minimum - two particles might give same error
        # best_particle_index = np.where(errors_champion == minimum_error)

        # # Initialise swarms best known position
        # swarm_best_position = swarm_position[:, best_particle_index]

        # return swarm_best_position, minimum_error

    def _update_historical_best(self):
        return None

    def _pso(self, X, y, no_particles, inertia, nostalgia, envy, upper, lower):
        """ Perform particle swarm optimisaion

        Arguments:
            X -- data matrix of size (no_examples, no_features)
            y -- response vector of size (no_examples, 1)

        Returns:
            w -- weights - a numpy vector of size (no_features, 1)
            b -- bias - a scalar

        """
        # add bias 
        X = self._add_bias(X)

        no_features_incl_bias, no_examples = X.shape
        
        # Initialise swarm position
        swarm_position = self._initialise_swarm(upper = upper, 
                                                lower = lower, 
                                                no_features = no_features_incl_bias,
                                                no_particles = no_particles)
        assert(swarm_position.shape == (no_features, no_particles))
        
        # Calculate errors for initial swarm position
        swarm_errors = self._predict_pso(X, swarm_position)
        assert(swarm_errors.shape == (no_features_incl_bias, no_particles))
        
        # Intialise particles best historical position and errors
        historical_best = swarm_position.copy()
        assert(historical_best.shape == (no_features, no_particles))
        historical_best_errors = swarm_errors.copy()
        
        # Update Swarms best historical position and errors   
        swarm_best_position, swarm_best_error = self._update_swarm_best(swarm_position,
                                                                        swarm_errors,
                                                                        historical_best,
                                                                        historical_best_errors)
        assert(swarm_best_position.shape == (no_features, 1))

        # Initialise Particles Velocity
        velocity = np.random.uniform(low = -math.abs(upper - lower), 
                                     high = math.abs(upper - lower), 
                                     size = (no_features, no_particles))

        #--------WHILE A TERMINATION CRITERIAN IS NOT MET
        for i in range(20):

            #Update Velocity
            velocity = self._update_velocity(velocity = velocity, 
                                             historical_best = historical_best, 
                                             swarm_best = swarm_best, 
                                             swarm_position = swarm_position,
                                             inertia = inertia,  
                                             nostalgia = nostalgia,
                                             envy = envy, 
                                             no_features = no_features,
                                             no_particles = no_particles)

            # Update particles position
            swarm_position = swarm_position + velocity

            # Update particles best known position
            predictions_challenger = self._predict_pso(X, swarm_position)
            assert(predictions_challenger.shape == (no_features_incl_bias, no_particles))

            errors_challenger= self._mae_pso(predictions_challenger, y)
            assert(errors_challenger.shape == (1, no_particles))

            historical_best, champion_error = self._update_historical_best(historical_best,
            )

            swarm_best, swarm_best_error = self._update_swarm_best()



        w = None
        b = None
        return w, b