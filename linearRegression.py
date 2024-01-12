def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0] 
    
    total_cost = 0
    
    cost_sum = 0
    for i in range(m):
        f_wb = (w * x[i]) + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    
    total_cost = (1 / (2 * m)) * cost_sum
    

    return total_cost




def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    m = x.shape[0]
    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = (w * x[i]) + b

        dj_dw_i = x[i] * (f_wb - y[i])

        dj_db_i = f_wb - y[i]

        dj_db += dj_db_i

        dj_dw += dj_dw_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m
        
    return dj_dw, dj_db