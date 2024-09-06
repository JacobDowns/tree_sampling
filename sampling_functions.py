import numpy as np
from numba import vectorize, float64, njit
import numpy.random as random

@njit
def propose_point(min_dist : float, max_dist : float, dx=1.):

    """
    Utility function that proposes a point between a minimum and maximum 
    distance from the origin.

    Parameters
    ----------
    min_radius : float
        Minimum distance of proposed point from origin.
    max_radius: float
        Maximum distance of proposed point from origin. 
    dx : float 
        Resolution of grid cell. 

    Returns
    -------
    (int, int)
        Relative grid offsets of proposed point. 
    """

    r = min_dist + random.random()*(max_dist - min_dist)
    theta =  2. * np.pi * random.random()
    di = r*np.cos(theta)
    dj = r*np.sin(theta)
    di = int(di / dx)
    dj = int(dj / dx)

    return (di, dj)


@njit 
def propose_index(i : int, j : int, ni : int, nj :int, dx=1., min_radius=1.5, max_radius=10., ):
    """
    Simple utility function that takes in indices on a grid and proposes new indices of a pixel on the grid 
    within an annulus. The proposed point is constrained to be on the grid. 

    Parameters
    ----------
    i : int
        Row index of initial point.
    j : int
        Column index of initial point
    ni : int 
        Number of rows on grid.
    nj : int
        Number of columns on grid.
    dx : float
        Resolution of grid cell.
    min_radius : float
        Minimum distance of proposed point from original points.
    max_radius: float
        Maximum distance of proposed point from original point.
  
    Returns
    -------
    (int, int)
        Grid indices of proposed point. 
    """
    
    # Propose new inddex
    di, dj = propose_point(min_radius, max_radius, dx)
    i1 = i + di 
    j1 = j + dj 

    # Make sure it's in bounds
    i1 = max(i1, 1)
    i1 = min(i1, ni-1)
    j1 = max(j1, 1)
    j1 = min(j1, nj-1)

    return (i1, j1)


@vectorize([float64(float64, float64, float64, float64, float64)])
def get_purves_radius(z, height, crown_base, dbh, trait_score):

    """
    Get radius at an array of z locations using the Purves crown model.

    Parameters
    ----------
    z : ndarray
        Array of z coordinates of float64 type. 
    height : float
        Tree height in meters.
    crown_base : float 
        Crown base in meters.
    dbh : float 
        Diameter at breast height in centimeters. 
    trait_score : float
        Purves shape parameter. 

    Returns
    -------
    r : ndarray
        Grid indices of proposed point. 
    """

    if z < crown_base:
        return 0.0
    
    if z > height:
        return 0.
    
    C0_R0 = 0.503
    C1_R0 = 3.126
    C0_R40 = 0.5
    C1_R40 = 10.0
    C0_B = 0.196
    C1_B = 0.511

    r0j = (1 - trait_score) * C0_R0 + trait_score * C1_R0
    r40j = (1 - trait_score) * C0_R40 + trait_score * C1_R40
    max_crown_radius = r0j + (r40j - r0j) * (dbh / 40.0)
    shape_parameter = (1 - trait_score) * C0_B + trait_score * C1_B

    return max_crown_radius * ((height - z) / height)**shape_parameter


@njit
def intersects(distance, props0, props1, scale=1.):

    """
    Check if two trees with a given distance and tree properties intersect. 

    Parameters
    ----------
    distance : float
        Distance between trunks of trees.
    props0 : ndarray
        Array of tree parameters of float type.
    props1 : ndarray
        Array of tree parameters of float type.
    scale : float 
        Rescale the radius of each tree. 

    Returns
    -------
    bool
        Returns True if trees intersect or false otherwise.
    """
    
    dbh0 = props0[0]
    height0 = props0[1]
    crown_ratio0 = props0[2]
    trait_score0 = props0[3]
    crown_base0 = height0 - crown_ratio0*height0

    dbh1 = props1[0]
    height1 = props1[1]
    crown_ratio1 = props1[2]
    trait_score1 = props1[3]
    crown_base1 = height1 - crown_ratio1*height1

    cb_max = max(crown_base0, crown_base1)
    r0_max = scale*get_purves_radius(cb_max, height0, crown_base0, dbh0, trait_score0)
    r1_max = scale*get_purves_radius(cb_max, height1, crown_base1, dbh1, trait_score1)

    return r0_max + r1_max >= distance


@njit
def accept(normalized_chm, tree_grid, tree_props, tree_index, i, j, dx=1., min_radius=1., radius=4., tree_scale=1., height_error=100.):
    """
    Given a proposed location for a tree on a grid, either accept or reject this placement. 

    Parameters
    ----------
    chm : ndarray
        Normalized crown height model of float type. 
    tree_props : ndarray
        Array of tree parameters of float type. Columns contain diameter at breast height, normalized height, crown radius, and trait score for each respectively.
    tree_grid : ndarray
        Array of tree indexes of int64 type. Zero values indicate no tree. Positive values refer to a particular tree index. 
    tree_index : int 
        Index of the proposed tree to place in the tree_props array. 
    i : int
        Proposed row index for tree
    j : int 
        Proposed column index for tree
    dx : float 
        Pixel size used to compute distances.
    min_radius : float
        Minimum distance between trees. Usually based on pixel size.
    radius : float
        The radius around the proposed tree location to check for conflicting trees. 
    tree_scale : float
        This variable rescale the radius of a tree, allowing for more or less tolerance for crown overlap.
    height_error : float
        Defines how much relative height difference we tolerate between the CHM and the proposed tree. 
        
    
    Returns
    -------
    bool
        Returns True if the tree placement is accepted or false otherwise.
    """
   
    # Only consider a point where thge normalized CHM is above a given threshold
    valid = normalized_chm[i,j] >= 1e-2 and tree_grid[i,j] == 0
    if not valid:
        return valid
    
    # Get properties of the tree we're placing
    props0 = tree_props[tree_index]
    
    # Is the relative height of this tree similar to the CHM?
    height = props0[1]
    if abs(height - normalized_chm[i,j]) > height_error:
        return False
    
    # Tree grid dimensions
    n0 = tree_grid.shape[0]
    n1 = tree_grid.shape[1]    
    
    # Check a local window around a tree to see if there's too much overlap with other trees
    w = int(radius / dx) + 1
    for i1 in range(max(1, i-w), min(n0-1, i+w)):
        for j1 in range(max(1, j-w), min(n1-1, j+w)):
            if tree_grid[i1,j1] > 0:
                
                # Make sure tree is further than minimum distance
                dist = np.sqrt(((i - i1)*dx)**2 +  ((j - j1)*dx)**2)
                if dist < min_radius:
                    return False
                
                # Get properties of this nearby tree
                props1 = tree_props[tree_grid[i1, j1] - 1]
                
                # Check for crown overlap
                collision = intersects(dist, props0, props1, tree_scale)
                if collision:
                    return False
                
    return True




@njit
def local_search(tree_grid, i, j, dx=1., max_dist=30.):
    """
    Performs a local search to find an unoccupied pixel near a point. 

    Parameters
    ----------
    tree_grid : ndarray
        Array of tree indexes of int64 type. Zero values indicate no tree. Positive values refer to a particular tree index. 
    i : int
        Proposed row index for tree
    j : int 
        Proposed column index for tree
    dx : float 
        Pixel size used to compute distances.
    max_dist : float
        Determines how large of an area to search. 
        
    Returns
    -------
    (int, int)
        Grid indices of free location or attempted indices.
    """
   
    
    # Tree grid dimensions
    n0 = tree_grid.shape[0]
    n1 = tree_grid.shape[1]    
    
    # Check a local window around a tree to see if there's an open spot
    i2 = i 
    j2 = j
    w = int(max_dist / dx) + 1
    for i1 in range(max(1, i-w), min(n0-1, i+w)):
        for j1 in range(max(1, j-w), min(n1-1, j+w)):
            i2 = i1 
            j2 = j1
            
            if tree_grid[i1,j1] == 0:                
               break

    return i2, j2


@njit 
def sample_trees(normalized_chm, tree_grid, tree_coords, tree_props, dx, tree_scale, max_dist, height_error):
    """
    Sample tree locations given a normalized CHM and a tree population. 

    Parameters
    ----------
    normalized_chm : ndarray
        Normalized crown height model of float type. 
    tree_grid : ndarray
        Array of tree indexes of int64 type. Zero values indicate no tree. Positive values refer to a particular tree index. 
    tree_coords : ndarray 
        Array of tree coordinates of type int refering to nearest pixel in CHM. First column is row index coordinate. Second is column index. 
    tree_props : ndarray
        Array of tree parameters of float type. Columns contain diameter at breast height, normalized height, crown radius, and trait score for each respectively.
    dx : float 
        Pixel size used to compute distances.
    scale : ndarray 
        1D array of floats that specifies how the radius of trees should be scaled per iteration of sampling. Allows for more crown overlap with smaller scales.
    max_dist : ndarray
        1D array of floats that specifies the maximum distance a tree can be placed from its original location on each iteration. 
    height_error : ndarray 
        1D array of floats that specifies the acceptable mismatch in relative CHM height versus relative tree height. 
        
    
    Returns
    -------
    tree_grid : ndarray
        Tree grid with tree indexes marked
    trees_to_place : list
        Returns the number of trees left to replace per iteration. Can be used to tune parameters. 
    """
    
    indexes = list(np.arange(len(tree_coords)))
    
    # This list will track the number of trees left to place at each iteration 
    trees_to_place = []

    # Iterate until we've placed all trees 
    for i in range(len(tree_scale)):
        new_indexes = []
        trees_to_place.append(len(indexes))
        
        for j in range(len(indexes)):
            tree_index = indexes[j]

            xi = tree_coords[tree_index,0]
            xj = tree_coords[tree_index,1]
            if i > 0:
                xi, xj = propose_index(xi, xj, tree_grid.shape[0], tree_grid.shape[1], dx, dx, max_dist[i])

            valid = accept(normalized_chm, tree_grid, tree_props, tree_index, xi, xj, dx, dx, 5., tree_scale[i], height_error[i])

            # If the tree's current position is good then place it on the grid
            if valid:
                tree_grid[xi, xj] = tree_index + 1
            else:
                # Otherwise add it to this list of indexes for subsequent passes
                new_indexes.append(tree_index)

        indexes = new_indexes
        if len(indexes) == 0:
            break
    
    # If there are some trees left over, just do a local search for an open space
    for j in range(len(indexes)):
        tree_index = indexes[j]
        xi = tree_coords[tree_index,0]
        xj = tree_coords[tree_index,1]
        
        xi, xj = local_search(normalized_chm, tree_grid, xi, xj, dx=dx, max_dist=30.)
        tree_grid[xi, xj] = tree_index + 1
        
    return tree_grid, trees_to_place
