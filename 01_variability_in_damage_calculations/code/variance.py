import numpy as np

def crit_dmg_multiplier(crit_amt):
    """
    Get the damage multiplier for landing a critical hit 

    inputs:
    crit_amt: int, critical hit stat

    returns:
    critical hit damage multiplier
    """
    return np.floor(200/3300 * (crit_amt - 380) + 1400)

def crit_prob(crit_amt):
    """
    Get the probability of landing a critical hit 

    inputs:
    crit_amt: int, critical hit stat

    returns:
    critical hit probability, from [0,1]
    """
    return np.floor(200/3300 * (crit_amt - 380) + 50) / 1000

def direct_hit_prob(dh_amt):
    """
    Get the probability of landing a direct hit 

    inputs:
    crit_amt: int, direct hit stat

    returns:
    direct hit probability, from [0,1]
    """
    return np.floor(550/3300 * (dh_amt - 380)) / 1000



def nh_supp(d2, buffs=None):
    """
    Find the support (all possible damage values) of a single normal hit, given D2 and any buffs

    inputs:
    d2: float, base D2 value
    buffs: lits/np array, list of buff multipliers. If none, no buffs will be used

    Returns:
    numpy array of the support for a single normal hit
    """

    if buffs is None:
        buff_prod = 1
    else:
        buff_prod = np.prod(buffs)

    lower, upper = np.floor(0.95 * d2), np.floor(1.05 * d2)+1
    damage_range = np.arange(lower, upper)

    support = np.floor(damage_range * buff_prod)
    return support

def ch_supp(d2, l_c, buffs=None):
    """
    Find the support (all possible damage values) of a single critical hit, given D2, the critical hit damage multiplier, and any buffs

    inputs:
    d2: float, base D2 value
    l_c: float, damage modifier for a critical hit. Value starts from 1000.
    buffs: lits/np array, list of buff multipliers. If none, no buffs will be used

    Returns:
    numpy array of the support for a single critical hit
    """
    if buffs is None:
        buff_prod = 1
    else:
        buff_prod = np.prod(buffs)

    ch_dmg = np.floor(np.floor(d2 * l_c) / 1000)
    lower, upper = np.floor(0.95 * ch_dmg), np.floor(1.05 * ch_dmg)+1
   
    damage_range = np.arange(lower, upper)

    support = np.floor(damage_range * buff_prod)
    return support

def dh_supp(d2, l_d, buffs=None):
    """
    Find the support (all possible damage values) of a single direct hit, given D2, the direct hit damage multiplier, and any buffs

    inputs:
    d2: float, base D2 value
    l_d: float, damage modifier for a direct hit. Value should be 125.
    buffs: lits/np array, list of buff multipliers. If none, no buffs will be used

    Returns:
    numpy array of the support for a single direct hit
    """
    if buffs is None:
        buff_prod = 1
    else:
        buff_prod = np.prod(buffs)

    dh_dmg = np.floor(np.floor(d2 * l_d) / 100)
    lower, upper = np.floor(0.95 * dh_dmg), np.floor(1.05 * dh_dmg)+1
   
    damage_range = np.arange(lower, upper)

    support = np.floor(damage_range * buff_prod)
    return support

def cdh_supp(d2, l_c, l_d, buffs=None):
    """
    Find the support (all possible damage values) of a single crit-direct hit, given D2, the hit-type damage multipliers, and any buffs

    inputs:
    d2: float, base D2 value
    l_c: float, damage modifier for a direct hit. Value starts from 1000.
    l_d: float, damage modifier for a direct hit. Value starts from 100.
    buffs: lits/np array, list of buff multipliers. If none, no buffs will be used

    Returns:
    numpy array of the support for a single crit-direct hit
    """
    if buffs is None:
        buff_prod = 1
    else:
        buff_prod = np.prod(buffs)
    
    ch_dmg = np.floor(np.floor(d2 * l_c) / 1000)
    cdh_dmg = np.floor(np.floor(ch_dmg * l_d) / 100)
    
    lower, upper = np.floor(0.95 * cdh_dmg), np.floor(1.05 * cdh_dmg)+1
   
    damage_range = np.arange(lower, upper)

    support = np.floor(damage_range * buff_prod)
    return support
    

# DoT supports are different because damage order of op. is uniform roll -> hit type -> buff
# Normal hits are the same for both cases though.
def dot_nh_supp(d2, buffs=None):
    return nh_supp(d2, buffs)

def dot_ch_supp(d2, l_c, buffs=None):
    """
    Find the support (all possible damage values) of a single critical hit for a DoT attack, given D2, the critical hit damage multiplier, and any buffs

    inputs:
    d2: float, base D2 value
    l_c: float, damage modifier for a critical hit. Value starts from 1000.
    buffs: lits/np array, list of buff multipliers. If none, no buffs will be used

    Returns:
    numpy array of the support for a single critical hit
    """
    if buffs is None:
        buff_prod = 1
    else:
        buff_prod = np.prod(buffs)

    lower, upper = np.floor(0.95 * d2), np.floor(1.05 * d2)+1
    damage_range = np.arange(lower, upper)

    support = np.floor(np.floor(np.floor(damage_range * l_c) / 1000) * buff_prod)
    return support

def dot_dh_supp(d2, l_d, buffs=None):
    """
    Find the support (all possible damage values) of a single direct hit of a DoT action, given D2, the direct hit damage multiplier, and any buffs

    inputs:
    d2: float, base D2 value
    l_d: float, damage modifier for a direct hit. Value should be 125.
    buffs: lits/np array, list of buff multipliers. If none, no buffs will be used

    Returns:
    numpy array of the support for a single direct hit
    """
    if buffs is None:
        buff_prod = 1
    else:
        buff_prod = np.prod(buffs)

    lower, upper = np.floor(0.95 * d2), np.floor(1.05 * d2)+1
    damage_range = np.arange(lower, upper)

    support = np.floor(np.floor(np.floor(damage_range * l_d) / 100) * buff_prod)
    return support

def dot_cdh_supp(d2, l_c, l_d, buffs=None):
    """
    Find the support (all possible damage values) of a single crit-direct hit, given D2, the hit-type damage multipliers, and any buffs

    inputs:
    d2: float, base D2 value
    l_c: float, damage modifier for a direct hit. Value starts from 1000.
    l_d: float, damage modifier for a direct hit. Value starts from 100.
    buffs: lits/np array, list of buff multipliers. If none, no buffs will be used

    Returns:
    numpy array of the support for a single crit-direct hit
    """
    if buffs is None:
        buff_prod = 1
    else:
        buff_prod = np.prod(buffs)

    lower, upper = np.floor(0.95 * d2), np.floor(1.05 * d2)+1
    damage_range = np.arange(lower, upper)

    ch_dmg = np.floor(np.floor(damage_range * l_c) / 1000)
    support = np.floor(np.floor(np.floor(ch_dmg * l_d) / 100) * buff_prod)
    return support


def hit_type_combos(n):
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



# Distribution bounds before buffs
#  Used for the sim functions
def get_nh_bounds(d2):
    return np.floor(0.95 * d2), np.floor(1.05 * d2) + 1

def get_ch_bounds(d2, l_c):
    crit_dmg = np.floor(np.floor(d2*l_c) / 1000)
    a_c = np.floor(0.95 * crit_dmg)
    b_c = np.floor(1.05 * crit_dmg) + 1

    return a_c, b_c

def get_dh_bounds(d2, l_d):
    avg_dh = np.floor(np.floor(d2*l_d) / 100)
    
    a_d = (np.floor(0.95 * avg_dh))
    b_d = (np.floor(1.05 * avg_dh) + 1)

    return a_d, b_d

def get_cdh_bounds(d2, l_c, l_d):

    avg_crit = np.floor(np.floor(d2*l_c) / 1000)
    avg_cdh = np.floor(np.floor(avg_crit * l_d) / 100)

    a_cd = (np.floor(0.95 * avg_cdh))
    b_cd = (np.floor(1.05 * avg_cdh) + 1)

    return a_cd, b_cd   