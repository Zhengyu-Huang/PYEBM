import numpy as np

#compute 2-D exact flux
def _exact_flux(prim_l, n_ij, eos):
    gamma = eos.gamma
    [rho,vx,vy,p] = prim_l

    E = 0.5*rho*(vx**2 + vy**2) + p/(gamma - 1.0)

    v_n = vx*n_ij[0] + vy*n_ij[1]

    return np.array([rho*v_n, rho*vx*v_n + p*n_ij[0] , rho*vy*v_n + p*n_ij[1], (E + p)*v_n],dtype=float)


def _wall_flux(prim_l, n_ij, eos):

    [rho,vx,vy,p] = prim_l




    return np.array([0.0, p*n_ij[0] , p*n_ij[1], 0.0],dtype=float)


# this is Fluid Fluid Roe Flux function,
# ww_l is left conservative variables
# ww_r is right conservative variables

def _Roe_flux(prim_l, prim_r, n_ij, eos):
    n_len = np.sqrt(n_ij[0]**2 + n_ij[1]**2)
    n_ij = n_ij/n_len
    gamma = eos.gamma


    t_ij = np.array([-n_ij[1],n_ij[0]],dtype = float)


    # left state
    [rho_l, u_l, v_l, p_l] = prim_l;
    un_l = u_l*n_ij[0] + v_l*n_ij[1]
    ut_l = u_l*t_ij[0] + v_l*t_ij[1]
    a_l = np.sqrt(gamma*p_l/rho_l)
    h_l = 0.5*(v_l*v_l + u_l*u_l) + gamma * p_l/(rho_l * (gamma - 1.0));

    # right state
    [rho_r, u_r, v_r, p_r] = prim_r;
    un_r = u_r*n_ij[0] + v_r*n_ij[1]
    ut_r = u_r*t_ij[0] + v_r*t_ij[1]
    a_r = np.sqrt(gamma*p_r/rho_r)
    h_r = 0.5*(v_r*v_r + u_r*u_r) + gamma * p_r/(rho_r * (gamma - 1.0));

    # compute the Roe-averaged quatities
    RT = np.sqrt(rho_r/rho_l)
    rho_rl = RT * rho_l

    u_rl = (u_l + RT*u_r)/(1.0 + RT)

    v_rl = (v_l + RT*v_r)/(1.0 + RT)
    h_rl = (h_l + RT*h_r)/(1.0 + RT)
    a_rl = np.sqrt((gamma - 1)*(h_rl - 0.5*(u_rl*u_rl + v_rl*v_rl)))
    un_rl = u_rl*n_ij[0] + v_rl*n_ij[1]
    ut_rl = u_rl*t_ij[0] + v_rl*t_ij[1]


    # wave strengths
    dp = p_r - p_l
    drho = rho_r - rho_l
    dun = un_r - un_l
    dut = ut_r -ut_l
    du = np.array([(dp - rho_rl*a_rl*dun)/(2.0*a_rl*a_rl),  rho_rl*dut,  drho - dp/(a_rl * a_rl),  (dp + rho_rl*a_rl*dun)/(2.0*a_rl*a_rl)],dtype=float)

    #compute the Roe-average wave speeds
    ws = np.array([np.fabs(un_rl - a_rl), np.fabs(un_rl), np.fabs(un_rl), np.fabs(un_rl + a_rl)],dtype = float)

    #Harten's Entropy Fix JCP(1983), 49, pp357-393:
    # only for the nonlinear fields.

    dws = 1.0/5.0
    if ( ws[0] < dws ):
        ws[0] = 0.50 * ( ws[0]*ws[0]/dws + dws )
    if ( ws[3] < dws ):
        ws[3] = 0.50 * ( ws[3]*ws[3]/dws + dws )


    #compute the right characteristic eigenvectors
    P_inv = np.array([[1.0,                     0.0,           1.0,                             1.0],
                       [u_rl - a_rl*n_ij[0],    t_ij[0],       u_rl ,                           u_rl + a_rl*n_ij[0]],
                       [v_rl - a_rl*n_ij[1],    t_ij[1] ,      v_rl ,                           v_rl + a_rl*n_ij[1]],
                       [h_rl - un_rl*a_rl,      ut_rl,         0.5*(u_rl*u_rl+v_rl*v_rl),       h_rl + un_rl*a_rl]],dtype=float)


    f_l = np.array([rho_l*un_l, rho_l*un_l*u_l + p_l*n_ij[0],  rho_l*un_l*v_l + p_l*n_ij[1], rho_l*h_l*un_l ])
    f_r = np.array([rho_r*un_r, rho_r*un_r*u_r + p_r*n_ij[0],  rho_r*un_r*v_r + p_r*n_ij[1], rho_r*h_r*un_r ])

    flux = 0.5*(f_r + f_l  - np.dot(P_inv, du*ws))



    return n_len*flux

'''
###########################################################################

def Pri_To_Conser(u,dim):
    global gamma

    if(dim == 3):
        [rho, v, p] = u;
        return  np.array([rho, rho*v, rho*v*v/2 + p/(gamma - 1)])
    elif(dim == 4):
        [rho, vx, vy, p] = u;
        return np.array([rho,rho*vx, rho*vy,rho*(vx*vx + vy*vy)/2 + p/(gamma - 1)])
    else:
        [rho, vx, vy, vz, p] = u;
        return np.array([rho,rho*vx, rho*vy,rho*vz, rho*(vx*vx + vy*vy + vz*vz)/2 + p/(gamma - 1)])


def Conser_To_Pri(w, dim):
    global gamma

    if(dim == 3):
        [w1, w2, w3] = w
        rho = w1;
        v = w2/w1;
        p = (w3 - w2*v/2) * (gamma - 1)
        return np.array([rho, v, p])
    elif(dim ==4):
        [w1, w2, w3, w4] = w
        rho = w1;
        vx = w2/w1;
        vy = w3/w1
        p = (w4 - w2*vx/2 - w3*vy/2) * (gamma - 1)
        return np.array([rho, vx, vy, p])
    else:

        [w1, w2, w3, w4,w5] = w
        rho = w1;
        vx = w2/w1;
        vy = w3/w1;
        vz = w4/w1;
        p = (w5 - w2*vx/2 - w3*vy/2 - w4*vz/2) * (gamma - 1)

        return np.array([rho, vx, vy,vz, p])


def FF_Roe_Flux(ww_i, ww_j, n,dim):

    global gamma

    if(dim == 4):
        n_len = np.sqrt(n[0]**2 + n[1]**2)
        nx,ny = n/n_len
        u_i = Conser_To_Pri(ww_i,4)
        u_j = Conser_To_Pri(ww_j,4)

        [rho_i, vx_i, vy_i,p_i] = u_i;
        [rho_j, vx_j, vy_j,p_j] = u_j;
        w_i = np.array([ww_i[0],ww_i[1],ww_i[2],0,ww_i[3]])
        w_j = np.array([ww_j[0],ww_j[1],ww_j[2],0,ww_j[3]])
        nz = 0.0;
        vz_i = 0.0
        vz_j = 0.0

    else:
        n_len = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        nx,ny,nz = n/n_len
        u_i = Conser_To_Pri(ww_i,5)
        u_j = Conser_To_Pri(ww_j,5)

        [rho_i, vx_i, vy_i, vz_i, p_i] = u_i;
        [rho_j, vx_j, vy_j, vz_j, p_j] = u_j;
        w_i =ww_i
        w_j = ww_j

    vn_i = vx_i*nx + vy_i*ny + vz_i*nz;
    H_i = gamma*p_i/(rho_i*(gamma - 1)) + 0.5*(vx_i**2 + vy_i**2 + vz_i**2)
    Fn_i = np.array([rho_i*vn_i, rho_i*vn_i*vx_i + p_i*nx,  rho_i*vn_i*vy_i + p_i*ny,rho_i*vn_i*vz_i + p_i*nz, rho_i*H_i*vn_i ])

    vn_j = vx_j*nx + vy_j*ny + vz_j*nz;
    H_j = gamma*p_j/(rho_j*(gamma - 1)) + 0.5*(vx_j**2 + vy_j**2 + vz_j**2)
    Fn_j = np.array([rho_j*vn_j, rho_j*vn_j*vx_j + p_j*nx,  rho_j*vn_j*vy_j + p_j*ny, rho_j*vn_j*vz_j + p_j*nz, rho_j*H_j*vn_j ])

    sr_i = np.sqrt(rho_i)
    sr_j = np.sqrt(rho_j)
    rho_RL = (sr_i*rho_i + sr_j*rho_j)/ (sr_i + sr_j)
    vx_RL  = (sr_i*vx_i + sr_j*vx_j)/(sr_i + sr_j)
    vy_RL  =  (sr_i*vy_i + sr_j*vy_j)/(sr_i + sr_j)
    vz_RL  =  (sr_i*vz_i + sr_j*vz_j)/(sr_i + sr_j)
    H_RL   =  (sr_i*H_i + sr_j*H_j)/(sr_i + sr_j)

    p_RL = rho_RL*(gamma - 1)/gamma*(H_RL - 0.5*(vx_RL**2 + vy_RL**2 + vz_RL**2))
    c_RL = np.sqrt(gamma*p_RL/rho_RL)

    v_RL = np.sqrt(vx_RL**2 + vy_RL**2 + vz_RL**2)
    vn_RL = vx_RL*nx + vy_RL*ny + vz_RL*nz

    vncross_RL = np.array([ny*vz_RL-nz*vy_RL, nz*vx_RL -nx*vz_RL, nx*vy_RL-ny*vx_RL])
    P_inv = np.array([[nx,             ny,             nz,        1/(2*c_RL**2),1/(2*c_RL**2)],
                       [vx_RL*nx,      vx_RL*ny - nz,       vx_RL*nz + ny, (vx_RL +c_RL*nx)/(2*c_RL**2), (vx_RL -c_RL*nx)/(2*c_RL**2)],
                       [vy_RL*nx +nz,     vy_RL*ny ,      vy_RL*nz - nx, (vy_RL +c_RL*ny)/(2*c_RL**2), (vy_RL -c_RL*ny)/(2*c_RL**2)],
                       [vz_RL*nx - ny, vz_RL*ny + nx,  vz_RL*nz,       (vz_RL +c_RL*nz)/(2*c_RL**2), (vz_RL -c_RL*nz)/(2*c_RL**2)],
                       [0.5*v_RL**2*nx - vncross_RL[0], 0.5*v_RL**2*ny - vncross_RL[1],  0.5*v_RL**2*nz - vncross_RL[2], (H_RL + c_RL*vn_RL)/(2*c_RL**2), (H_RL - c_RL*vn_RL)/(2*c_RL**2)]])

    P = np.array([  [nx - (gamma - 1)*v_RL**2*nx/(2*c_RL**2) + vncross_RL[0], nx*(gamma - 1)*vx_RL/(c_RL**2),  nz + nx*(gamma-1)*vy_RL/c_RL**2,  -ny + nx*(gamma-1)*vz_RL/c_RL**2,  -nx*(gamma - 1)/c_RL**2],
                    [ny - (gamma - 1)*v_RL**2*ny/(2*c_RL**2) + vncross_RL[1], -nz + ny*(gamma - 1)*vx_RL/(c_RL**2),  ny*(gamma-1)*vy_RL/c_RL**2,  nx + ny*(gamma-1)*vz_RL/c_RL**2,  -ny*(gamma-1)/c_RL**2 ],
                    [nz - (gamma - 1)*v_RL**2*nz/(2*c_RL**2) + vncross_RL[2], ny + nz*(gamma - 1)*vx_RL/(c_RL**2), -nx +  nz*(gamma-1)*vy_RL/c_RL**2,  nz*(gamma-1)*vz_RL/c_RL**2,  -nz*(gamma-1)/c_RL**2 ],
                    [(gamma-1)*v_RL**2/2 - c_RL*vn_RL,        c_RL*nx - (gamma - 1)*vx_RL,     c_RL*ny -(gamma-1)*vy_RL, c_RL*nz -(gamma-1)*vz_RL,   gamma-1],
                    [(gamma-1)*v_RL**2/2 + c_RL*vn_RL,       -c_RL*nx - (gamma - 1)*vx_RL,     -c_RL*ny -(gamma-1)*vy_RL,  -c_RL*nz -(gamma-1)*vz_RL , gamma-1]])

    D = np.diag([vn_RL,vn_RL,vn_RL, vn_RL+c_RL,vn_RL-c_RL])
    Dabs = np.diag([abs(vn_RL), abs(vn_RL), abs(vn_RL), abs(vn_RL+c_RL),abs(vn_RL-c_RL)])



    roe_flux = 0.5*(Fn_i + Fn_j) - 0.5*np.dot(P_inv,np.dot(Dabs,np.dot(P,w_j - w_i)))


    if(dim == 4):
        roe_flux = np.array([roe_flux[0],roe_flux[1],roe_flux[2],roe_flux[4]])

    return roe_flux*n_len

#Roe flux check

gamma = 1.4
u_l = np.array([1.,2.,2.,5])
u_r = np.array([6.,3.,3.,6])

w_l = Pri_To_Conser(u_l,4)
w_r = Pri_To_Conser(u_r,4)

n =np.array([2.,1.])
n = n/np.linalg.norm(n)

print FF_Roe_Flux(w_l,w_r,n,4)
print _Roe_flux(u_l,u_r,n,gamma)
#on the shock w is not


n *= 2.0
n =np.array([1.0,1.0])
n = n/np.linalg.norm(n)
u_l = np.array([1.,2.,77.,5])
u_r = np.array([6.,3.,66.,6])

w_l = Pri_To_Conser(u_l,4)
w_r = Pri_To_Conser(u_r,4)
print FF_Roe_Flux(w_l,w_r, n,4)

print _Roe_flux(u_l,u_r,n,gamma)

##############################################
gamma = 1.4
n =np.array([1.0,1.0])
n = n/np.linalg.norm(n)
u_l = np.array([1.,np.sqrt(3.0), 0.0, 1/1.4])
u_r = np.array([1.,np.sqrt(3.0), -1.0,1/1.4])

w_l = Pri_To_Conser(u_l,4)
w_r = Pri_To_Conser(u_r,4)
print FF_Roe_Flux(w_l,w_r, n,4)
print _Roe_flux(u_l,u_r,n,gamma)

#############################################
'''










# Compute steger warming matrix P+ and P-
# here P = k(0)A + k(1)B
def _Steger_Warming(prim_l, W_oo, k, eos):
    gamma = eos.gamma

    #k = k/np.linalg.norm(k)
    k_norm = np.linalg.norm(k)
    [tilde_kx,tilde_ky] = k/k_norm
    [rho,vx,vy,p] = prim_l


    v = np.dot([vx,vy],k)
    c = np.sqrt(gamma * p/ rho)


    Dp = np.array([max(0,v), max(0,v), max(0,v+c*k_norm),max(0,v-c*k_norm)],dtype=float)
    Dm = np.array([min(0,v), min(0,v), min(0,v+c*k_norm),min(0,v-c*k_norm)],dtype=float)

    theta = tilde_kx*vx + tilde_ky*vy

    phi = np.sqrt((gamma - 1)/2*(vx*vx + vy*vy))

    beta = 1/(2*c*c)

    Q = np.array([[1,0,1,1],
                 [vx,tilde_ky,vx + tilde_kx*c, vx - tilde_kx*c],
                 [vy,-tilde_kx, vy+tilde_ky*c, vy-tilde_ky*c],
                 [phi*phi/(gamma-1),tilde_ky*vx-tilde_kx*vy, (phi*phi + c*c)/(gamma-1)+c*theta,(phi*phi + c*c)/(gamma-1)-c*theta]])

    Qinv = np.array([[1 - phi*phi/(c*c), (gamma - 1)*vx/c**2, (gamma - 1)*vy/c**2,-(gamma - 1)/c**2],
                    [-(tilde_ky*vx - tilde_kx*vy),tilde_ky,-tilde_kx,0],
                    [beta*(phi**2 - c*theta), beta*(tilde_kx*c - (gamma - 1)*vx), beta*(tilde_ky*c - (gamma - 1)*vy),beta*(gamma-1)],
                    [beta*(phi**2 + c*theta), -beta*(tilde_kx*c + (gamma - 1)*vx), -beta*(tilde_ky*c + (gamma - 1)*vy),beta*(gamma-1)]])

    fp = 0.5*rho/gamma * np.array([2.0*(gamma-1)*Dp[0] + Dp[2] + Dp[3],
                                    2.0*(gamma-1)*Dp[0]*vx + Dp[2]*(vx + c*tilde_kx) + Dp[3]*(vx - c*tilde_kx),
                                    2.0*(gamma-1)*Dp[0]*vy + Dp[2]*(vy + c*tilde_ky) + Dp[3]*(vy - c*tilde_ky),
                                    (gamma-1)*Dp[0]*(vx*vx + vy*vy) + 0.5*Dp[2]*((vx+c*tilde_kx)**2 + (vy+c*tilde_ky)**2) + 0.5*Dp[3]*((vx - c*tilde_kx)**2 + (vy -c*tilde_ky)**2)
                                    + (3.0 - gamma)*(Dp[2] + Dp[3])*c*c/(2*(gamma-1))], dtype=float)

    fm = np.dot(Q,  Dm * np.dot(Qinv,W_oo))



    return fp + fm






'''
# Compute steger warming matrix P+ and P-
# here P = k(0)A + k(1)B
def _Steger_Warming(prim_l, W_oo, k, eos):
    gamma = eos.gamma

    #k = k/np.linalg.norm(k)
    k_norm = np.linalg.norm(k)
    [tilde_kx,tilde_ky] = k/k_norm
    [rho,vx,vy,p] = prim_l


    v = np.dot([vx,vy],k)
    c = np.sqrt(gamma * p/ rho)


    Dp = np.array([max(0,v), max(0,v), max(0,v+c*k_norm),max(0,v-c*k_norm)],dtype=float)
    Dm = np.array([min(0,v), min(0,v), min(0,v+c*k_norm),min(0,v-c*k_norm)],dtype=float)

    theta = tilde_kx*vx + tilde_ky*vy

    phi = np.sqrt((gamma - 1)/2*(vx*vx + vy*vy))

    beta = 1/(2*c*c)

    Q = np.array([[1,0,1,1],
                 [vx,tilde_ky,vx + tilde_kx*c, vx - tilde_kx*c],
                 [vy,-tilde_kx, vy+tilde_ky*c, vy-tilde_ky*c],
                 [phi*phi/(gamma-1),tilde_ky*vx-tilde_kx*vy, (phi*phi + c*c)/(gamma-1)+c*theta,(phi*phi + c*c)/(gamma-1)-c*theta]])

    Qinv = np.array([[1 - phi*phi/(c*c), (gamma - 1)*vx/c**2, (gamma - 1)*vy/c**2,-(gamma - 1)/c**2],
                    [-(tilde_ky*vx - tilde_kx*vy),tilde_ky,-tilde_kx,0],
                    [beta*(phi**2 - c*theta), beta*(tilde_kx*c - (gamma - 1)*vx), beta*(tilde_ky*c - (gamma - 1)*vy),beta*(gamma-1)],
                    [beta*(phi**2 + c*theta), -beta*(tilde_kx*c + (gamma - 1)*vx), -beta*(tilde_ky*c + (gamma - 1)*vy),beta*(gamma-1)]])

    fp = 0.5*rho/gamma * np.array([2.0*(gamma-1)*Dp[0] + Dp[2] + Dp[3],
                                    2.0*(gamma-1)*Dp[0]*vx + Dp[2]*(vx + c*tilde_kx) + Dp[3]*(vx - c*tilde_kx),
                                    2.0*(gamma-1)*Dp[0]*vy + Dp[2]*(vy + c*tilde_ky) + Dp[3]*(vy - c*tilde_ky),
                                    (gamma-1)*Dp[0]*(vx*vx + vy*vy) + 0.5*Dp[2]*((vx+c*tilde_kx)**2 + (vy+c*tilde_ky)**2) + 0.5*Dp[3]*((vx - c*tilde_kx)**2 + (vy -c*tilde_ky)**2)
                                    + (3.0 - gamma)*(Dp[2] + Dp[3])*c*c/(2*(gamma-1))], dtype=float)

    fm = np.dot(Q,  Dm * np.dot(Qinv,W_oo))

'''


def _Riemann_bWstar_FS(V,vv_wall,nn_wall,eos,equation_type):
    """
    This is Piston Riemann Problem

    Args:
       V: fluid state primitive variables rho, u_x, u_y, p, at the wall, float[4]
       vv_wall: wall velocity, float[3]
       nn_wall: woll outward normal, normal direction is away from fluid float[3]
       eos: equation of state
       equation_type: Euler or NavierStokes

    Returns:

    """


    nn_wall = nn_wall/np.linalg.norm(nn_wall)
    gamma = eos.gamma;
    [rho, vx, vy, p] = V;

    vn = vx*nn_wall[0] + vy*nn_wall[1]

    vn_wall = vv_wall[0]*nn_wall[0] + vv_wall[1]*nn_wall[1]

    if(vn > vn_wall):
        #print "Shock Wave FSRiemann"
        a = 2/((gamma + 1)*rho);
        b = p*(gamma - 1)/(gamma + 1)
        phi = a/(vn_wall - vn)**2

        p_R = p + (1 + np.sqrt(4*phi*(p + b) + 1))/(2*phi)
        rho_R = rho*(p_R/p + (gamma - 1)/(gamma + 1))/(p_R/p * (gamma - 1)/(gamma + 1) + 1)

    else:
        #print "Rarefactions FSRiemann"
        c = np.sqrt(gamma*p/rho);
        p_R = p*(-(gamma - 1)/(2*c)*(vn_wall - vn) + 1)**(2*gamma/(gamma - 1))
        rho_R = rho*(p_R/p)**(1/gamma);

    if(equation_type == "Euler"):
        return np.array([rho_R, (vn_wall - vn)*nn_wall[0] + vx, (vn_wall -vn)*nn_wall[1] + vy, p_R],dtype=float)
    else:
        return np.array([rho_R, vv_wall[0], vv_wall[1], p_R],dtype=float)
