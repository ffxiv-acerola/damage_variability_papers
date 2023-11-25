import numpy as np
from scipy.stats import multinomial, skewnorm
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

class Support():

    def __init__(self, d2, l_c, is_dot, buffs=None, l_d=None) -> None:
        """
        Class to compute the support of a single hit from an action.

        inputs:
        d2 - int, base damage of an action.
        l_c - int, modifier for landing a critical hit.
        is_dot - bool, whether the action is a DoT effect (has different support than direct dmg).
        buffs - list, List of any buffs present. A 10% damage buff would be `[1.10]`. 
                If no buffs are present, then an empty list `[]`, list with none (`[None]`), or `[1]` can be supplied
        l_d - int, modifier for landing a direct hit. Not currently used, but might be for auto direct hit skills.
        """
        if l_d is None:
            self.l_d = 125

        self.d2 = d2
        self.l_c = l_c
        self.buff_prod = self.buff_check(buffs)
        self.is_dot = is_dot

        self.normal_supp = self.get_support('normal')
        self.crit_supp = self.get_support('critical')
        self.dir_supp = self.get_support('direct')
        self.crit_dir_supp = self.get_support('critical-direct')
        pass

    def buff_check(self, buffs):
        if buffs is None or len(buffs) == 0:
            return 1
        else:
            return np.product(buffs)

    def ch_dmg_modifier(self):
        """
        Damage modifier for landing a critical hit.
        """
        return np.floor(np.floor(self.d2 * self.l_c) / 1000)
    
    def dh_dmg_modifier(self):
        """
        Damage modifier for landing a direct hit.
        """
        return np.floor(np.floor(self.d2 * self.l_d) / 100)

    def cdh_dmg_modifier(self):
        """
        Damage modifier for landing a critical-direct hit.
        """
        ch_damage = np.floor(np.floor(self.d2 * self.l_c) / 1000)
        return np.floor(np.floor(ch_damage * self.l_d) / 100)

    def get_support(self, hit_type):
        """
        Find the support (all possible damage values) of a single hit, given D2 and any buffs

        input:
        hit_type - str, type of hit. Can be `normal`, `critical`, `direct`, or `critical-direct`

        Returns:
        numpy array of the support for a single normal hit
        """
        if self.is_dot:
            lower, upper = np.floor(0.95 * self.d2), np.floor(1.05 * self.d2)+1
            damage_range = np.arange(lower, upper)
            
            if hit_type == 'normal':
                support = np.floor(damage_range * self.buff_prod)

            elif hit_type == 'critical':
                support = np.floor(np.floor(np.floor(damage_range * self.l_c) / 1000) * self.buff_prod)

            elif hit_type == 'direct':
                support = np.floor(np.floor(np.floor(damage_range * self.l_d) / 100) * self.buff_prod)
            
            elif hit_type == 'critical-direct':
                ch_dmg = np.floor(np.floor(damage_range * self.l_c) / 1000)
                support = np.floor(np.floor(np.floor(ch_dmg * self.l_d) / 100) * self.buff_prod)
            
            else:
                print('incorrect input')

        # Attack is not a DoT
        else:
            if hit_type == 'normal':
                lower, upper = np.floor(0.95 * self.d2), np.floor(1.05 * self.d2)+1

            elif hit_type == 'critical':
                lower, upper = np.floor(0.95 * self.ch_dmg_modifier()), np.floor(1.05 * self.ch_dmg_modifier())+1   

            elif hit_type == 'direct':
                lower, upper = np.floor(0.95 * self.dh_dmg_modifier()), np.floor(1.05 * self.dh_dmg_modifier())+1

            elif hit_type == 'critical-direct':
                lower, upper = np.floor(0.95 * self.cdh_dmg_modifier()), np.floor(1.05 * self.cdh_dmg_modifier())+1

            damage_range = np.arange(lower, upper)
            support = np.floor(damage_range * self.buff_prod)

        return support


class ActionMoments(Support):

    def __init__(self, action_df, t) -> None:
        """
        Compute moments for a action landing n_hits

        inputs: 
        action_df - pandas dataframe with the schema {d2} 
        """

        self.n = action_df['n']
        self.p = action_df['p']
        self.t = t
        if 'action-name' in action_df:
            self.action_name = action_df['action-name']
        # All possible hit types
        self.x = self.hit_type_combos(self.n)
        # Corresponding multinomial weight
        self.w = multinomial(self.n, self.p).pmf(self.x)

        Support.__init__(self, action_df['d2'], action_df['l_c'], bool(action_df['is-dot']), action_df['buffs'])
        
        # Lots of notation for computing moments when there are gaps
        self._S_N = self.normal_supp.size
        self._Z_N = np.sum(self.normal_supp)
        self._Z_N2 = np.sum(self.normal_supp**2)
        self._Z_N3 = np.sum(self.normal_supp**3)

        self._S_C = self.crit_supp.size
        self._Z_C = np.sum(self.crit_supp)
        self._Z_C2 = np.sum(self.crit_supp**2)
        self._Z_C3 = np.sum(self.crit_supp**3)

        self._S_D = self.dir_supp.size
        self._Z_D =  np.sum(self.dir_supp)
        self._Z_D2 = np.sum(self.dir_supp**2)
        self._Z_D3 = np.sum(self.dir_supp**3)

        self._S_CD = self.crit_dir_supp.size
        self._Z_CD =  np.sum(self.crit_dir_supp)
        self._Z_CD2 = np.sum(self.crit_dir_supp**2)
        self._Z_CD3 = np.sum(self.crit_dir_supp**3)

        self._first_moment = self._get_first_moment()
        self._second_moment = self._get_second_moment()
        self._third_moment = self._get_third_moment()

        self.mean = self._first_moment
        self.variance = self.get_action_variance()
        self.skewness = self.get_action_skewness()

        # Convert from total damage to DPS
        self.mean /= self.t
        self.variance /= self.t**2

        pass

    def hit_type_combos(self, n):
        """
        This will give all of the different hit combinations and will sum to n 
        For example, if N = 10, some of the combinations are
        [0, 5, 5, 10]
        [10, 0, 0, 0]
        [3, 1, 3, 3]

        and so on
        
        Idk how it works because I copy/pasted from stackoverflow, but my God is it fast compared to nested loops
        https://stackoverflow.com/questions/34970848/find-all-combination-that-sum-to-n-with-multiple-lists
        """
        import itertools, operator
        hit_list = []
        for cuts in itertools.combinations_with_replacement(range(n+1), 3):
            combi = list(map(operator.sub, cuts + (n,), (0,) + cuts))
            if max(combi) < n:
                hit_list.append(combi)

        # The loop doesn't include cases where there a n types of just 1 hit type
        # Hardcoding it in is easy
        hit_list.append([n, 0, 0, 0])
        hit_list.append([0, n, 0, 0])
        hit_list.append([0, 0, n, 0])
        hit_list.append([0, 0, 0, n])
        return np.array(hit_list)

    def _get_first_moment(self):
        """
        Compute the first moment (mean) for an action landing n hits.
        """
        first_deriv = ((self.x[:,1]*self._Z_C)/self._S_C + (self.x[:,3]*self._Z_CD)/self._S_CD + (self.x[:,2]*self._Z_D)/self._S_D + (self.x[:,0]*self._Z_N)/self._S_N)
        return np.dot(self.w, first_deriv)

    def _get_second_moment(self):
        """
        Compute the second moment for an action landing n hits
        """
        second_deriv = ((self.x[:,1]**2*self._Z_C**2)/self._S_C**2 + (self.x[:,3]**2*self._Z_CD**2)/self._S_CD**2 \
                        + self.x[:,3]*(-(self._Z_CD**2/self._S_CD**2) + self._Z_CD2/self._S_CD) + (self.x[:,2]**2*self._Z_D**2)/self._S_D**2 \
                        + self.x[:,1]*(-(self._Z_C**2/self._S_C**2) + self._Z_C2/self._S_C + (2*self.x[:,3]*self._Z_C*self._Z_CD)/(self._S_C*self._S_CD) \
                        + (2*self.x[:,2]*self._Z_C*self._Z_D)/(self._S_C*self._S_D)) + self.x[:,2]*((2*self.x[:,3]*self._Z_CD*self._Z_D)/(self._S_CD*self._S_D) \
                        - self._Z_D**2/self._S_D**2 + self._Z_D2/self._S_D) + (self.x[:,0]**2*self._Z_N**2)/self._S_N**2 \
                        + self.x[:,0]*((2*self.x[:,1]*self._Z_C*self._Z_N)/(self._S_C*self._S_N) + (2*self.x[:,3]*self._Z_CD*self._Z_N)/(self._S_CD*self._S_N)\
                        + (2*self.x[:,2]*self._Z_D*self._Z_N)/(self._S_D*self._S_N) - self._Z_N**2/self._S_N**2 + self._Z_N2/self._S_N))
        return np.dot(self.w, second_deriv)

    def _get_third_moment(self):
        """
        Compute the third moment for an action landing n hits.
        """

        third_deriv = ((self.x[:,1]**3*self._Z_C**3)/self._S_C**3 + (self.x[:,3]**3*self._Z_CD**3)/self._S_CD**3 \
                        + self.x[:,3]**2*((-3*self._Z_CD**3)/self._S_CD**3 + (3*self._Z_CD*self._Z_CD2)/self._S_CD**2) \
                        + self.x[:,3]*((2*self._Z_CD**3)/self._S_CD**3 - (3*self._Z_CD*self._Z_CD2)/self._S_CD**2 + self._Z_CD3/self._S_CD) \
                        + (self.x[:,2]**3*self._Z_D**3)/self._S_D**3 + self.x[:,1]**2*((-3*self._Z_C**3)/self._S_C**3 \
                        + (3*self._Z_C*self._Z_C2)/self._S_C**2 + (3*self.x[:,3]*self._Z_C**2*self._Z_CD)/(self._S_C**2*self._S_CD) 
                        + (3*self.x[:,2]*self._Z_C**2*self._Z_D)/(self._S_C**2*self._S_D)) + self.x[:,2]**2*((3*self.x[:,3]*self._Z_CD*self._Z_D**2)/(self._S_CD*self._S_D**2) \
                        - (3*self._Z_D**3)/self._S_D**3 + (3*self._Z_D*self._Z_D2)/self._S_D**2) + self.x[:,1]*((2*self._Z_C**3)/self._S_C**3 \
                        - (3*self._Z_C*self._Z_C2)/self._S_C**2 + self._Z_C3/self._S_C + (3*self.x[:,3]**2*self._Z_C*self._Z_CD**2)/(self._S_C*self._S_CD**2) \
                        + self.x[:,3]*((-3*self._Z_C**2*self._Z_CD)/(self._S_C**2*self._S_CD) + (3*self._Z_C2*self._Z_CD)/(self._S_C*self._S_CD) \
                        - (3*self._Z_C*self._Z_CD**2)/(self._S_C*self._S_CD**2) + (3*self._Z_C*self._Z_CD2)/(self._S_C*self._S_CD)) + (3*self.x[:,2]**2*self._Z_C*self._Z_D**2)/(self._S_C*self._S_D**2) \
                        + self.x[:,2]*((-3*self._Z_C**2*self._Z_D)/(self._S_C**2*self._S_D) + (3*self._Z_C2*self._Z_D)/(self._S_C*self._S_D) + (6*self.x[:,3]*self._Z_C*self._Z_CD*self._Z_D)/(self._S_C*self._S_CD*self._S_D)\
                        - (3*self._Z_C*self._Z_D**2)/(self._S_C*self._S_D**2) + (3*self._Z_C*self._Z_D2)/(self._S_C*self._S_D))) + self.x[:,2]*((3*self.x[:,3]**2*self._Z_CD**2*self._Z_D)/(self._S_CD**2*self._S_D) \
                        + (2*self._Z_D**3)/self._S_D**3 - (3*self._Z_D*self._Z_D2)/self._S_D**2 + self.x[:,3]*((-3*self._Z_CD**2*self._Z_D)/(self._S_CD**2*self._S_D) \
                        + (3*self._Z_CD2*self._Z_D)/(self._S_CD*self._S_D) - (3*self._Z_CD*self._Z_D**2)/(self._S_CD*self._S_D**2) + (3*self._Z_CD*self._Z_D2)/(self._S_CD*self._S_D)) + self._Z_D3/self._S_D) \
                        + (self.x[:,0]**3*self._Z_N**3)/self._S_N**3 + self.x[:,0]**2*((3*self.x[:,1]*self._Z_C*self._Z_N**2)/(self._S_C*self._S_N**2) \
                        + (3*self.x[:,3]*self._Z_CD*self._Z_N**2)/(self._S_CD*self._S_N**2) + (3*self.x[:,2]*self._Z_D*self._Z_N**2)/(self._S_D*self._S_N**2) \
                        - (3*self._Z_N**3)/self._S_N**3 + (3*self._Z_N*self._Z_N2)/self._S_N**2) + self.x[:,0]*((3*self.x[:,1]**2*self._Z_C**2*self._Z_N)/(self._S_C**2*self._S_N) \
                        + (3*self.x[:,3]**2*self._Z_CD**2*self._Z_N)/(self._S_CD**2*self._S_N) + (3*self.x[:,2]**2*self._Z_D**2*self._Z_N)/(self._S_D**2*self._S_N) \
                        + (2*self._Z_N**3)/self._S_N**3 - (3*self._Z_N*self._Z_N2)/self._S_N**2 + self.x[:,1]*((-3*self._Z_C**2*self._Z_N)/(self._S_C**2*self._S_N) \
                        + (3*self._Z_C2*self._Z_N)/(self._S_C*self._S_N) + (6*self.x[:,3]*self._Z_C*self._Z_CD*self._Z_N)/(self._S_C*self._S_CD*self._S_N) + (6*self.x[:,2]*self._Z_C*self._Z_D*self._Z_N)/(self._S_C*self._S_D*self._S_N) \
                        - (3*self._Z_C*self._Z_N**2)/(self._S_C*self._S_N**2) + (3*self._Z_C*self._Z_N2)/(self._S_C*self._S_N)) + self.x[:,3]*((-3*self._Z_CD**2*self._Z_N)/(self._S_CD**2*self._S_N) \
                        + (3*self._Z_CD2*self._Z_N)/(self._S_CD*self._S_N) - (3*self._Z_CD*self._Z_N**2)/(self._S_CD*self._S_N**2) + (3*self._Z_CD*self._Z_N2)/(self._S_CD*self._S_N)) \
                        + self.x[:,2]*((6*self.x[:,3]*self._Z_CD*self._Z_D*self._Z_N)/(self._S_CD*self._S_D*self._S_N) - (3*self._Z_D**2*self._Z_N)/(self._S_D**2*self._S_N) + (3*self._Z_D2*self._Z_N)/(self._S_D*self._S_N) \
                        - (3*self._Z_D*self._Z_N**2)/(self._S_D*self._S_N**2) + (3*self._Z_D*self._Z_N2)/(self._S_D*self._S_N)) + self._Z_N3/self._S_N))

        return np.dot(self.w, third_deriv)
    
    def get_action_variance(self):
        return self._second_moment - self.mean**2

    def get_action_skewness(self):
        return (self._third_moment - 3*self.mean*self.variance - self.mean**3) / self.variance**(3./2.)

class Rotation():

    def __init__(self, rotation_df, t) -> None:
        
        self.rotation_df = rotation_df
        self.t = t
        self.action_moments = [ActionMoments(row, t) for _, row in rotation_df.iterrows()]
        self.action_names = rotation_df['action-name'].tolist()
        self.action_means = np.array([x.mean for x in self.action_moments]) 
        self.action_variances = np.array([x.variance for x in self.action_moments])
        self.action_std = np.sqrt(self.action_variances)
        self.action_skewness = np.array([x.skewness for x in self.action_moments]) 

        self.rotation_mean = np.sum(self.action_means)
        self.rotation_variance = np.sum(self.action_variances)
        self.rotation_std = np.sqrt(self.rotation_variance)
        # Need just the numerator of Pearson's skewness, which is why we multiply by the action variances inside the sum
        self.rotation_skewness = np.sum(self.action_skewness * self.action_variances**(3/2)) / np.sum(self.action_variances)**(3/2) 

        self.compute_DPS_distributions()

        pass

    def compute_DPS_distributions(self) -> None:
        """
        Compute and set the support and PMF of DPS distributions.

        This method is broken into 3 sections
        (i) Individual actions (remember Action A with Buff 1 is distinct from Action A with Buff 2).
        (ii) Unique actions (Action A with Buff 1 and Action A with Buff 2 are group together now).
        (iii) The entire rotation.
        """

        # DPS is discretized by this much.
        # Bigger number = faster but larger discretization error
        # Smaller number = slower but more accurate.
        delta = 0.5

        # section (i)
        self.action_dps_support = [None] * self.action_means.size
        self.action_DPS_distributions = [None] * self.action_means.size
        self.rotation_DPS_distribution = None

        skew_norm_indices = []
        convolve_indices = []
        low_high_rolls = np.zeros((self.action_means.size,2)) # lowest and highest possible damage values
        # First check if DPS distribution should be computed via convolution or parameterize a Skew normal dist.
        for a in range(self.action_means.size):
            # TODO: More robust switchover from convolutions -> skew norm
            # include values of p in this check
            # Will need more testing
            n = self.action_moments[a].n
            low_high_rolls[a,:] = ([int(n * self.action_moments[a].normal_supp[0] / self.t), 
                                    int(n * self.action_moments[a].crit_dir_supp[-1] / self.t)])
            if self.action_moments[a].n < 20:
                x, y = self.convolve_pmf(a)
                self.action_dps_support[a] = np.arange(low_high_rolls[a,0], low_high_rolls[a,1] + delta, step=delta)
                self.action_DPS_distributions[a] = np.interp(self.action_dps_support[a], x, y)
                convolve_indices.append(a)
            else:
                skew_norm_indices.append(a)
                alpha, omega, squigma = self.moments_to_skew_norm(self.action_means[a], self.action_variances[a], self.action_skewness[a])
                self.action_dps_support[a] = np.arange(np.floor(low_high_rolls[a,0]),
                                                      np.floor(low_high_rolls[a,1]) + delta, step=delta)
                self.action_DPS_distributions[a] = skewnorm.pdf(self.action_dps_support[a], alpha, squigma, omega)

        # Section (ii)
        self.unique_actions = {n: [] for n in {x.split('-')[0] for x in self.action_names}}
        [self.unique_actions[k].append(idx) for k in self.unique_actions.keys() for idx, s in enumerate(self.action_names) if k in s]
        self.unique_actions_distribution = {}
        
        for _, (name, action_idx_list) in enumerate(self.unique_actions.items()):
            action_low_high = np.zeros((len(self.unique_actions[name]), 2))

            # Support is sum of all lowest possible value (min rol NH) to highest possible value (max roll CDH)
            for idx, action_idx in enumerate(action_idx_list):
                action_low_high[idx, :] = np.array([self.action_dps_support[action_idx].min(), self.action_dps_support[action_idx].max()])
            
            support = np.arange(action_low_high[:,0].sum(), action_low_high[:,1].sum() + delta, step=delta)

            if len(action_idx_list) == 1:
                action_dps_distribution = self.action_DPS_distributions[action_idx_list[0]]

            elif len(action_idx_list) > 1:
                action_dps_distribution = fftconvolve(self.action_DPS_distributions[action_idx_list[0]], 
                                                    self.action_DPS_distributions[action_idx_list[1]])

            if len(action_idx_list) > 2:
                for idx in range(1, len(action_idx_list)-1):
                    action_dps_distribution = fftconvolve(action_dps_distribution, 
                                                          self.action_DPS_distributions[action_idx_list[idx+1]])

            self.unique_actions_distribution[name] = {'support': support, 'dps_distribution': action_dps_distribution}


        # Section (iii)
        self.rotation_dps_support = np.arange(low_high_rolls[:,0].sum(), low_high_rolls[:,1].sum() + delta, step=delta)
        
        if len(self.action_moments) > 1:
            self.rotation_DPS_distribution = fftconvolve(self.action_DPS_distributions[0], self.action_DPS_distributions[1])
        else:
            self.rotation_DPS_distribution = self.action_DPS_distributions[0]

        if len(self.action_moments) > 2:
            for a in range(2, len(self.action_DPS_distributions)):
                self.rotation_DPS_distribution = fftconvolve(self.action_DPS_distributions[a], self.rotation_DPS_distribution)            
        
        # Need to renormalize the DPS distribution
        self.rotation_DPS_distribution *= delta**(len(self.action_DPS_distributions) - 1)
        pass

    @classmethod
    def moments_to_skew_norm(self, mean, variance, skewness):
        """
        Converts the mean, variance, and Pearson's skewness to parameters defined by skew normal.
        The parameters are not the same, but can be interconverted: https://en.wikipedia.org/wiki/Skew_normal_distribution
        """

        delta = np.sqrt(np.pi/2 * (np.abs(skewness))**(2/3) / (np.abs(skewness)**(2/3)+((4-np.pi)/2)**(2/3)))
        alpha = np.sign(skewness) * delta / np.sqrt(1 - delta**2)
        omega  = np.sqrt(variance / (1 - 2*delta**2 / np.pi))
        squigma = mean - omega * delta * np.sqrt(2/np.pi)

        return alpha, omega, squigma

    def convolve_pmf(self, action_idx):
        """
        Convolve the single-hit PMF of a action n_hit times to get the exact PMF of a action landing n_hits
        This isn't done for every action because it becomes slow as n_hits gets large. 
        This should only be done when a skew-normal distribution does not fit the actual DPS PMF.
        (Which happens when n_hits is small or elements of **p** are near 0 or 1)
        
        Inputs:
        action_idx
        """

        def multi_conv(pmf, n_hits):
            if n_hits == 1:
                return pmf
            else:
                conv_pmf = fftconvolve(pmf, pmf)
                for _ in range(n_hits-2):
                    conv_pmf = fftconvolve(conv_pmf, pmf)
                return conv_pmf

        # make a shorter variable name cause this long
        action_moment = self.action_moments[action_idx]

        # Define the bounds of the mixture distribution (lowest roll NH and highest roll CDH)
        # Everything is integers, so the bounds can be defined with an arange
        min_roll = np.floor(action_moment.normal_supp[0]).astype(int)
        max_roll = np.floor(action_moment.crit_dir_supp[-1]).astype(int)

        self.one_hit_pmf = np.zeros(max_roll - min_roll + 1)

        # Need to find out how many indices away the start of each hit-type subdistribution is from
        # the lower bound of the mixture distribution.
        ch_offset = int(action_moment.crit_supp[0] - action_moment.normal_supp[0])
        dh_offset = int(action_moment.dir_supp[0] - action_moment.normal_supp[0])
        cdh_offset = int(action_moment.crit_dir_supp[0] - action_moment.normal_supp[0])

        # Set up slices to include gaps
        normal_slice = (action_moment.normal_supp - action_moment.normal_supp[0]).astype(int)
        ch_slice = (action_moment.crit_supp - action_moment.crit_supp[0] + ch_offset).astype(int)
        dh_slice = (action_moment.dir_supp - action_moment.dir_supp[0] + dh_offset).astype(int)
        cdh_slice = (action_moment.crit_dir_supp - action_moment.crit_dir_supp[0] + cdh_offset).astype(int)

        # Mixture distribution defined with multinomial weights
        self.one_hit_pmf[normal_slice] = action_moment.p[0] / action_moment.normal_supp.size
        self.one_hit_pmf[ch_slice] =  action_moment.p[1] / action_moment.crit_supp.size
        self.one_hit_pmf[dh_slice] = action_moment.p[2] / action_moment.dir_supp.size
        self.one_hit_pmf[cdh_slice] = action_moment.p[3] / action_moment.crit_dir_supp.size

        conv_pmf = multi_conv(self.one_hit_pmf, action_moment.n)
        lowest_roll = int(np.floor(action_moment.normal_supp[0])*action_moment.n)

        # Convert from damage to DPS
        dmg_supp = np.arange(lowest_roll, conv_pmf.size + lowest_roll, step=1) / self.t
        # Multiplying by t ensures the PMF still sums to 1
        conv_pmf = conv_pmf * self.t

        return dmg_supp, conv_pmf

    def plot_action_distributions(self, ax=None, **kwargs):
        """
        Plot DPS distributions for each unique action.
        Recall a action is unique based on the action *and* the buffs present.
        Action A with Buff 1 is different than Action A with Buff 2.

        inputs:
        ax - matplotlib axis object, optional axis object to plot onto. 
             If one is not supplied, a new figure and shown at the end.
        **kwargs - any kwargs to be passed to, `ax.plot(**kwargs)`.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5,4), dpi=150)
            return_ax = False

        else:
            return_ax = True
        
        for a in range(self.action_means.size):
            ax.plot(self.action_dps_support[a], self.action_DPS_distributions[a], label=self.action_names[a])
        ax.set_xlabel('Damage per Second (DPS)')

        if return_ax:
            return ax
        else:
            plt.show()
            pass

    def plot_unique_action_distribution(self, ax=None, **kwargs):
        """
        Plot DPS distribution for unique actions, grouped by action name.
        For example, this would should the sum of DPS distributions for Action A with Buff 1 and Action A with Buff 2
        and label it as "Action A".

        inputs:
        ax - matplotlib axis object, optional axis object to plot onto. 
             If one is not supplied, a new figure and shown at the end.
        **kwargs - any kwargs to be passed to, `ax.plot(**kwargs)`.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5,4), dpi=150)
            return_ax = False

        else:
            return_ax = True

        for _, (name, distributions) in enumerate(self.unique_actions_distribution.items()):
            ax.plot(distributions['support'], distributions['dps_distribution'], label=name, **kwargs)

        ax.legend()
        if return_ax:
            return ax
        else:
            plt.show()
            pass

    def plot_rotation_distribution(self, ax=None, **kwargs):
        """
        Plot the overall DPS distribution for the rotation.

        inputs:
        ax - matplotlib axis object, optional axis object to plot onto. 
             If one is not supplied, a new figure and shown at the end.
        **kwargs - any kwargs to be passed to, `ax.plot(**kwargs)`.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5,4), dpi=150)
            return_ax = False

        else:
            return_ax = True

        alpha, omega, squigma = self.moments_to_skew_norm(self.rotation_mean, self.rotation_variance, self.rotation_skewness)
        x = np.linspace(self.rotation_mean - 5 * self.rotation_std, self.rotation_mean + 5 * self.rotation_std, 100)
        y = skewnorm.pdf(x, alpha, squigma, omega)

        ax.plot(x, y, **kwargs)
        ax.set_xlabel('Damage per Second (DPS)')

        if return_ax:
            return ax
        else:
            plt.show()
            pass

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    from counting import Rate
    from moments import Rotation
    a = Rate(2500, 1500)

    # t = 2.5 * 20
    rotation = pd.DataFrame({
        'potency': [2500, 5000, 5000],
        'n': [5, 15, 10],
        'p': [a.p, a.p, a.p],
        'l_c': [a.l_c, a.l_c, a.l_c],
        'buffs': [[1.1], [1.1], [1.1, 1.2]],
        'is-dot': [0, 0, 1],
        'action-name': ['test', 'test', 'test'],
        'infusion-adjust': [299, 299, 299]
    })
    
    # r = Rotation(rotation, t)
    # r.plot_action_distributions()
    # r.plot_rotation_distribution()

    # t = 2.5*20
    # rotation = pd.DataFrame({
    #     'd2': [100],
    #     'n': [1],
    #     'p': [a.p],
    #     'l_c': [a.l_c],
    #     'buffs': [1.1],
    #     'is-dot': [0],
    #     'action-name': ['Glare']})

    t = 112
    rotation = pd.DataFrame({
        'd2': [25000, 32000, 9500, 5300],
        'n': [44, 3, 4, t//3],
        'p': [a.p]*4,
        'l_c': [a.l_c]*4,
        'buffs': [1]*4,
        'is-dot': [0, 0, 0, 1],
        'action-name': ['Glare', 'Assize', 'Dia (application)', 'Dia (tick)']})

    t = 112
    rotation = pd.DataFrame({
        'd2': [25000],
        'n': [5],
        'p': [a.p],
        'l_c': [a.l_c],
        'buffs': [1],
        'is-dot': [0],
        'action-name': ['Glare']})

    rotation = pd.DataFrame({
        'd2': [6000, 1375],
        'n': [133, 120],
        'p': [a.p]*2,
        'l_c': [a.l_c]*2,
        'buffs': [None, None],
        'is-dot': [0, 1],
        'action-name': ['Fall Malefic', 'Combust']})

    rotation = pd.DataFrame({
        'd2': [6000, 6000, 1375]*2,
        'n': [133, 133, 120]*2,
        'p': [a.p]*6,
        'l_c': [a.l_c]*6,
        'buffs': [None]*6,
        'is-dot': [0, 0, 1]*2,
        'action-name': ['Fall Malefic', '', 'Combust']*2})

    r = Rotation(rotation, t)
    r.convolve_pmf(0)
    r.plot_action_distributions()
    r.plot_rotation_distribution()
    print('done')