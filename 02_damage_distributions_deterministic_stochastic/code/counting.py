import numpy as np

class Rate():

    def __init__(self, crit_amt, dh_amt) -> None:
        """
        Get probabilities of different hit types given critical hit and direct hit rate stats.
        """
        self.crit_amt = crit_amt
        self.dh_amt = dh_amt
        self.p = self.get_p()
        self.l_c = self.crit_dmg_multiplier()
        pass

    def crit_dmg_multiplier(self) -> float:
        """
        Get the damage multiplier for landing a critical hit 

        inputs:
        crit_amt: int, critical hit stat

        returns:
        critical hit damage multiplier
        """
        return np.floor(200/1900 * (self.crit_amt - 400) + 1400)

    def crit_prob(self) -> float:
        """
        Get the probability of landing a critical hit 

        inputs:
        crit_amt: int, critical hit stat

        returns:
        critical hit probability, from [0,1]
        """
        return np.floor(200/1900 * (self.crit_amt - 400) + 50) / 1000

    def direct_hit_prob(self) -> float:
        """
        Get the probability of landing a direct hit 

        inputs:
        crit_amt: int, direct hit stat

        returns:
        direct hit probability, from [0,1]
        """
        return np.floor(550/1900 * (self.dh_amt - 400)) / 1000

    def get_p(self, ch_mod=0, dh_mod=0, keep_cd=True):
        """
        Get the probability of each hit type occurring given the probability of a critical hit and direct hit

        inputs:
        crit_amt: Amount of critical hit stat.
        dh_amt: Amount of direct hit rate stat.
        ch_mod: Percentage to increase the base critical hit rate by if buffs are present.
                E.g., ch_mod=0.1 would add 0.1 to the base critical hit rate   
        dh_mod: Percentage to increase the base direct hit rate by if buffs are present.
        keep_cd: bool, Exploratory argument to see what happens critical-direct hits are removed.
                 THIS SHOULD ALWAYS BE TRUE IF FOR IN GAME CALCULATIONS. 
                 If true, the probability is p_c*p_d, otherwise 0

        returns:
        probability of each hit type, [normal hit, critical hit given not CD hit, direct hit given not CDH hit, CDH hit]
        """

        # Sometimes the probabilities don't sum to exactly 1.0 because of floating point error.
        # Sometimes SciPy's multinomial class will break because of that and just return NaN for probabilities.
        # Using the Decimal class with arbitrary precision and then converting them back to a float remedies this I guess.
        # Floating point math is stupid.
        from decimal import Decimal
        p_c = Decimal(self.crit_prob() + ch_mod)
        p_d = Decimal(self.direct_hit_prob() + dh_mod)
        if keep_cd:
            p_cd = Decimal(p_c) * Decimal(p_d)
        
        else:
            p_cd = Decimal(0)

        return np.array([float(Decimal(1.0) - p_c - p_d + p_cd), float(p_c - p_cd), float(p_d - p_cd), float(p_cd)])


