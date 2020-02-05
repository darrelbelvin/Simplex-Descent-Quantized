import numpy as np

def simplex_descent_quantized(func_quanta, quanta, x0, max_iters = 1000, **kwargs):
    dims = x0.shape[0]
    exit_cause = None

    if type(quanta) != np.array:
        quanta = np.ones(x0.shape) * quanta

    max_step = None
    if 'max_step' in kwargs:
        max_step = kwargs['max_step']

    bounds = None
    if 'bounds' in kwargs:
        bounds = kwargs['bounds']
        # bounds should be array-like minimums in first column, maximums in second

        if type(bounds) != np.ndarray:
            bounds = np.array(bounds)
        
        if np.any(bounds[1] < bounds[0]):
            raise ValueError("'bounds' maximum values (second column) must be greater than minimum (first column)")
        
        # convert to quanta
        bounds = (bounds / quanta).astype(int)

        # func_quanta = bounded_func(func_quanta, bounds)
        # todo: make bounded_func

    basin_mapping = False
    if 'basin_mapping' in kwargs:
        basin_mapping = kwargs['basin_mapping']
    
    current_basin_map = None
    this_map = None
    if basin_mapping:
        if bounds is None:
            raise ValueError("'bounds' is required if 'basin_mapping' is true")
        
        if 'current_basin_map' in kwargs:
            current_basin_map = kwargs['current_basin_map']
        
        if current_basin_map is not None:
            this_map = np.zeros_like(current_basin_map, dtype=bool)
        else:
            this_map = np.zeros(bounds[:,1]-bounds[:,0], dtype=bool)
    
    func = lambda x_quanta: func_quanta(x_quanta * quanta)

    evals={}

    def get_point_bounded(new_point):
        if bounds is not None:
            # under = new_point < bounds[...,0]
            # over = new_point > bounds[...,1]
            # if np.any(under) or np.any(over): # reflect off the boundary
            #     new_point[under] -= (new_point[under] - bounds[0, under])*2
            #     new_point[over] -= (new_point[over] - bounds[1, over])*2
            new_point = np.minimum(np.maximum(new_point, bounds[...,0]), bounds[...,1])
        return new_point

    def get_val_bounded(new_point):
        

        listed = tuple(new_point)
        if listed in evals.keys():
            return evals[listed]
        
        val = func(new_point)
        evals[listed] = val
        return val

    def step(new_point, new_val, reorder = True):
        nonlocal simplex, values, iter_count
        iter_count += 1

        simplex[-1] = new_point
        values[-1] = new_val
        
        if reorder:
            order = np.argpartition(values, (0, -1),)

            simplex[:] = simplex[order]
            values[:] = values[order]
            # worst two are now in the -2 and -1 position and best is in the 0 position
        
        if basin_mapping:
            try:
                M_inv = np.linalg.inv(simplex[1:] - simplex[0])

                mn = np.min(simplex, axis=0)
                mx = np.max(simplex, axis=0)
                box = this_map[tuple([slice(n, x) for n, x in zip(mn - bounds[:,0], mx - bounds[:,0])])]
                
                ind = np.array(list(np.ndindex(box.shape))).reshape(np.concatenate([box.shape, [dims]])) + mn
                delta = ind - simplex[0]

                delta_sim = delta @ M_inv

                mask = np.all(delta_sim >= 0, axis=-1)
                mask = mask & (np.sum(delta_sim, axis=-1) <= 1)
                box[mask] = True

            except np.linalg.LinAlgError:
                print('simplex is degenerate')
            
            

    def exit_check():
        nonlocal simplex, values, iter_count, exit_cause
        if current_basin_map is not None:
            if current_basin_map[tuple(simplex[0])] != 0:
                # best is in an existing basin
                current_basin_map[this_map] = current_basin_map[tuple(simplex[0])]
                exit_cause = "basin found"
                return True
                    
        #print(iter_count, simplex, values)
        if np.all((np.max(simplex, axis=0) - np.min(simplex, axis=0)) <= dims**0.5):
            exit_cause = "maximum found"
            current_basin_map[this_map] = np.max(current_basin_map) + 1
            return True

        if iter_count >= max_iters:
            exit_cause = "max_iters"
            return True
        
        return False
    
    
    if 'initial_simplex' in kwargs:
        simplex = np.array(kwargs['initial_simplex'] / quanta).astype(int)
    else:
        simplex = x0 + np.concatenate([np.identity(dims, dtype=int) * 10, np.zeros((1,dims), dtype=int)]) - 2
        simplex = np.apply_along_axis(get_point_bounded, 1, simplex)
    assert simplex.shape == (dims + 1, dims)

    values = np.array([func(x) for x in simplex])

    iter_count = 0

    step(simplex[-1], func(simplex[-1]))

    while not exit_check():

        if values[-1] < values[0]: # we are currently expanding
            force_stop = False
            if max_step is not None and np.linalg.norm(worst_hat) * (1+mult) > max_step:
                beyond = np.round(simplex[-1] + worst_hat * np.linalg.norm(worst_hat) / max_step).astype(int)
                can_expand = True
            else:
                beyond = np.round(simplex[-1] + worst_hat * (1+mult)).astype(int)

            beyond_val = get_val_bounded(beyond)

            if beyond_val < values[-1]: #it's even better, step and keep expanding
                step(beyond, beyond_val, force_stop)
                mult *= 2
            else:
                step(simplex[-1], values[-1]) # this will reorder simplex and stop expansion
            continue

        worst_hat = np.mean(simplex[:-1], axis=0) - simplex[-1]

        beyond = np.round(simplex[-1] + worst_hat * 2).astype(int)
        beyond_val = get_val_bounded(beyond)

        if values[0] <= beyond_val <= values[-2]:
            # not an extreme, step
            step(beyond, beyond_val)
        
        elif beyond_val < values[0]:
            # it's the best, expand
            mult = 2
            step(beyond, beyond_val, False) # not reordering triggers expansion

        else: # np.any(np.abs(worst_hat) > 1):
            # still the worst, shrink
            delta = np.round(worst_hat * 0.501).astype(int)
            
            shrunken = simplex[-1] + delta
            shrunken_val = get_val_bounded(shrunken)

            if np.all(shrunken == simplex[-1]):
                print('not moving')
                break
            
            step(shrunken, shrunken_val)

    #order = np.argpartition(values, (0, -1),)

    #simplex[:] = simplex[order]
    #values[:] = values[order]
    # import matplotlib.pyplot as plt
    # plt.imshow(this_map)
    # plt.show()

    return {'x':simplex[0],
            'y':values[0],
            'exit_cause': exit_cause,
            'n_iter': iter_count,
            'new_basins': current_basin_map}


if __name__ == "__main__":
    ndim=2
    coef1 = np.random.randint(-100,100,ndim)
    coef2 = np.random.randint(1,24,ndim)/25

    bounds1 = [[-100,100],[-100,100]]
    result = simplex_descent_quantized(lambda x: np.sum(x*coef1 + (x**2)*coef2),
         1, np.zeros(ndim), 100, bounds=bounds1, basin_mapping=True)

    # result = simplex_descent_quantized(lambda x: np.sum(x*coef1 + (x**2)*coef2), 1, np.zeros(ndim), 10000)
    print(result)