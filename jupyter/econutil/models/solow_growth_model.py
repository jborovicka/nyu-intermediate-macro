# functions for plotting the solow growth model output for a range of exercises

# import necessary libraries - everything collected in defs_intermediate_macro
import numpy as np
import econutil as ec

# ==============================================================================
# function definitions

# ------------------------------------------------------------------------------
# steady state calculation from model parameters
def Solow_steady_state(param):
    ss = dict()
    ss['K'] = (param['s']*param['A']/param['δ'])**(1/(1-param['α']))*param['L']
    ss['Y'] = param['A'] * ss['K']**param['α'] * param['L']**(1-param['α'])
    ss['I'] = param['s']*ss['Y']
    ss['C'] = (1-param['s'])*ss['Y']
    ss['r'] = param['α']*ss['Y']/ss['K']
    ss['w'] = (1-param['α'])*ss['Y']/param['L']
    ss['y'] = ss['Y']/param['L']
    ss['k'] = ss['K']/param['L']
    return ss

# ------------------------------------------------------------------------------
# plot the Solow diagram: p are model parameters, auxp are auxiliary parameters
def Solow_diagram(p, auxp=dict()):
    ss = Solow_steady_state(p)

    # default definitions for the graph
    K_max = 1.5*ss['K']
    fig_param = {'figsize' : [10,6], 'fontsize': 16,
                 'title': '',
                 'xlim': [0,K_max], 'ylim': [0,1.1*ss['Y']],
                 'xlabel': 'capital $K$', 'ylabel': '',
                 'ylogscale': False,
                 'showgrid': True, 'highlightzero': False,
                 'showNBERrecessions': True, 'showNBERrecessions_y': [11000,20000],
                 'plot_graph': True}
    
    # overwrite keys in p using param
    for i in auxp.keys():
        fig_param[i] = auxp[i]

    if fig_param['plot_graph']:
        fig,ax = ec.GenerateTSPlot(fig_param)

        K_grid = np.linspace(0,K_max,200)
        Y_grid = p['A']*K_grid**p['α']*p['L']**(1-p['α'])

        ax.plot(K_grid,Y_grid, linewidth=3,linestyle='solid',marker='',color=ec.clist_vibrant[0],label='output $Y$')
        ax.plot(K_grid,p['s']*Y_grid, linewidth=3,linestyle='dashed',marker='',color=ec.clist_vibrant[1],label='investment $I$')
        ax.plot(K_grid,p['δ']*K_grid, linewidth=3,linestyle='dashdot',marker='',color=ec.clist_vibrant[4],label='depreciation $\\delta K$')
        # ax.plot(K_grid,(1-param['s'])*Y_grid, linewidth=3,marker='',color=clist_1[3],label='consumption $C$')

        # plot axis labels
        cur_xticks = ax.get_xticks()
        cur_xlim = ax.get_xlim()
        cur_yticks = ax.get_yticks()
        cur_ylim = ax.get_ylim()

        #ax.set_xticks(list(cur_xticks) + [ss['K']])
        #ax.set_xticklabels(list(cur_xticks) + ['$K^*$'])
        ax.set_xticks([ss['K']])
        ax.set_xticklabels(['$K^*$'])
        ax.set_xlim(cur_xlim)

        #ax.set_yticks(list(np.round(cur_yticks,1)) + [ss['I'],ss['Y']])
        #ax.set_yticklabels(list(np.round(cur_yticks,1)) + ['$I^* = \delta K^*$','$Y^*$'])
        ax.set_yticks([ss['I'],ss['Y']]) # ,ss['C']
        ax.set_yticklabels(['$I^* = \\delta K^*$','$Y^*$']) # ,'$C^*$'
        ax.set_ylim([0,1.05*max(Y_grid)])

        ax.legend(loc='upper left')
    else:
        fig = None
        ax = None

    data = dict()
    data['fig'] = fig
    data['ax'] = ax
    data['K_grid'] = K_grid
    data['Y_grid'] = Y_grid
    data['param'] = p
    data['ss'] = ss

    return data

# ------------------------------------------------------------------------------
# plot the Solow diagram: pold, pnew are model parameters, auxp are auxiliary parameters
def Solow_diagram_experiment(p_old, p_new, auxp=dict()):

    ss_old = Solow_steady_state(p_old)
    ss_new = Solow_steady_state(p_new)

    # definitions for the graph
    K_max = 1.3*np.maximum(ss_old['K'],ss_new['K'])
    fig_param = {'figsize' : [15,9], 'fontsize': 16,
                 'title': '',
                 'xlim': [0,K_max], 'ylim': [0,1.1*np.maximum(ss_old['Y'],ss_new['Y'])],
                 'xlabel': 'capital $K$', 'ylabel': '',
                 'ylogscale': False,
                 'showgrid': True, 'highlightzero': False,
                 'showNBERrecessions': True, 'showNBERrecessions_y': [11000,20000],
                 'plot_graph': True}

    # overwrite keys in p using param
    for i in auxp.keys():
        fig_param[i] = auxp[i]

    if fig_param['plot_graph']:
        fig,ax = ec.GenerateTSPlot(fig_param)

        K_grid = np.linspace(0,K_max,200)
        Y_old_grid = p_old['A']*K_grid**p_old['α']*p_old['L']**(1-p_old['α'])
        Y_new_grid = p_new['A']*K_grid**p_new['α']*p_new['L']**(1-p_new['α'])

        ax.plot(K_grid,Y_old_grid, linewidth=2,marker='',linestyle='--',alpha=0.5,color=ec.clist_vibrant[0],label='output $Y_{old}$')
        ax.plot(K_grid,p_old['s']*Y_old_grid, linewidth=2,marker='',linestyle='--',alpha=0.5,color=ec.clist_vibrant[1],label='investment $I_{old}$')
        ax.plot(K_grid,p_old['δ']*K_grid, linewidth=2,marker='',linestyle='--',alpha=0.5,color=ec.clist_vibrant[4],label='depreciation $\\delta_{old} K$')

        ax.plot(K_grid,Y_new_grid, linewidth=3,marker='',alpha=0.5,color=ec.clist_vibrant[0],label='output $Y_{new}$')
        ax.plot(K_grid,p_new['s']*Y_new_grid, linewidth=3,marker='',alpha=0.5,color=ec.clist_vibrant[1],label='investment $I_{new}$')
        ax.plot(K_grid,p_new['δ']*K_grid, linewidth=3,marker='',alpha=0.5,color=ec.clist_vibrant[4],label='depreciation $\\delta_{new} K$')

        # plot axis labels
        cur_xticks = ax.get_xticks()
        cur_xlim = ax.get_xlim()
        cur_yticks = ax.get_yticks()
        cur_ylim = ax.get_ylim()

        ax.set_xticks([p_new['K0'],ss_new['K'],ss_old['K']])
        ax.set_xticklabels(['$K_0$','$K^*_{new}$','$K^*_{old}$'])
        
        #ax.set_xticks([p_new['K0'],ss_new['K']])
        #ax.set_xticklabels(['$K_0$','$K^*_{new}$'])
        ax.set_xlim(cur_xlim)

        ax.set_yticks([ss_new['I'],ss_new['Y'],ss_old['I'],ss_old['Y']])
        ax.set_yticklabels(['$I^*_{new} = \\delta_{new} K^*_{new}$','$Y^*_{new}$','$I^*_{old} = \\delta_{old} K^*_{old}$','$Y^*_{old}$'])
        ax.set_ylim(cur_ylim)

        ax.legend(loc='upper left');
    else:
        fig = None
        ax = None

    data = dict()
    data['fig'] = fig
    data['ax'] = ax
    data['K_grid'] = K_grid
    data['Y_old_grid'] = Y_old_grid
    data['Y_new_grid'] = Y_new_grid
    data['p_old'] = p_old
    data['p_new'] = p_new
    data['ss_old'] = ss_old
    data['ss_new'] = ss_new

    return data

# ------------------------------------------------------------------------------
# plot the trajectories: pold, pnew are model parameters, auxp are auxiliary parameters
def Solow_trajectories_experiment(p_old, p_new, auxp=dict()):

    ss_old = Solow_steady_state(p_old)
    ss_new = Solow_steady_state(p_new)

    # definitions for the graph
    Tmin, Tmax = -5, 100
    T_path = np.linspace(Tmin,Tmax,Tmax-Tmin+1)
    
    fig_param = {'figsize' : [12,12], 'fontsize': 16, 'subplots': [4,2],
                 'title': '',
                 'xlim': [Tmin,Tmax], 'ylim': [1,1],
                 'xlabel': 'time $t$', 'ylabel': '',
                 'ylogscale': False,
                 'showgrid': True, 'highlightzero': False,
                 'showNBERrecessions': True, 'showNBERrecessions_y': [11000,20000],
                 'trajectories': ['K','Y','I','C','r','w','k','y'],
                 'plot_graph': True}
    
    # overwrite keys in p using param
    for i in auxp.keys():
        fig_param[i] = auxp[i]
    
    # compute trajectories
    K_path = np.empty(Tmax-Tmin+1)

    # path starts in old steady state for periods Tmin,...,0
    K_path[0:-Tmin] = ss_old['K']
    K_path[-Tmin] = p_new['K0']

    for t in range(-Tmin,Tmax-Tmin):
        K_path[t+1] = (1-p_new['δ'])*K_path[t] + p_new['s']*p_new['A']*K_path[t]**p_new['α']*p_new['L']**(1-p_new['α'])

    Y_path = p_new['A'] * K_path**p_new['α'] * p_new['L']**(1-p_new['α'])
    I_path = p_new['s'] * Y_path
    C_path = (1-p_new['s']) * Y_path
    r_path = p_new['α']*Y_path/K_path
    w_path = (1-p_new['α'])*Y_path/p_new['L']
    k_path = K_path/p_new['L']
    y_path = Y_path/p_new['L']
    
    K_0 = K_path[-Tmin]
    Y_0 = Y_path[-Tmin]
    I_0 = I_path[-Tmin]
    C_0 = C_path[-Tmin]
    r_0 = r_path[-Tmin]
    w_0 = w_path[-Tmin]
    k_0 = k_path[-Tmin]
    y_0 = y_path[-Tmin]

    Y_path[0:-Tmin] = p_old['A'] * K_path[0:-Tmin]**p_old['α'] * p_old['L']**(1-p_old['α'])
    I_path[0:-Tmin] = p_old['s'] * Y_path[0:-Tmin]
    C_path[0:-Tmin] = (1-p_old['s']) * Y_path[0:-Tmin]
    r_path[0:-Tmin] = p_old['α']*Y_path[0:-Tmin]/K_path[0:-Tmin]
    w_path[0:-Tmin] = (1-p_old['α'])*Y_path[0:-Tmin]/p_old['L']
    k_path[0:-Tmin] = K_path[0:-Tmin]/p_old['L']
    y_path[0:-Tmin] = Y_path[0:-Tmin]/p_old['L']

    if fig_param['plot_graph']:
        fig,ax = ec.GenerateTSPlot(fig_param)

        for i in range(fig_param['subplots'][0]):
            for j in range(fig_param['subplots'][1]):
            
                if i*fig_param['subplots'][1]+j >= len(fig_param['trajectories']):
                    ax[i][j].axis('off')
                elif fig_param['trajectories'][i*fig_param['subplots'][1]+j] == 'K':
                    # capital path
                    ylim = [min(K_path)-0.1*(max(K_path)-min(K_path)),max(K_path)+0.1*(max(K_path)-min(K_path))]
                    ax[i][j].plot(T_path[0:-Tmin],K_path[0:-Tmin], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[4],label='capital $K_t$')
                    ax[i][j].plot(T_path[-Tmin-1:-Tmin+1],K_path[-Tmin-1:-Tmin+1], linewidth=1,marker='',linestyle=':',alpha=1,color=ec.clist_vibrant[4])
                    ax[i][j].plot(T_path[-Tmin:Tmax-Tmin+1],K_path[-Tmin:Tmax-Tmin+1], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[4])
                    ax[i][j].plot([0,0],ylim,linewidth=2,linestyle=':',color='k')
                    ax[i][j].set_ylabel('capital $K_t$')
                    ax[i][j].set_yticks([K_0,ss_new['K'],ss_old['K']])
                    ax[i][j].set_yticklabels(['$K_0$','$K^*_{new}$','$K^*_{old}$'])
                    ax[i][j].set_ylim(ylim)
                elif fig_param['trajectories'][i*fig_param['subplots'][1]+j] == 'Y':
                    # output path
                    ylim = [min(Y_path)-0.1*(max(Y_path)-min(Y_path)),max(Y_path)+0.1*(max(Y_path)-min(Y_path))]
                    ax[i][j].plot(T_path[0:-Tmin],Y_path[0:-Tmin], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[0],label='output $Y_t$')
                    ax[i][j].plot(T_path[-Tmin-1:-Tmin+1],Y_path[-Tmin-1:-Tmin+1], linewidth=1,marker='',linestyle=':',alpha=1,color=ec.clist_vibrant[0])
                    ax[i][j].plot(T_path[-Tmin:Tmax-Tmin+1],Y_path[-Tmin:Tmax-Tmin+1], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[0])
                    ax[i][j].plot([0,0],ylim,linewidth=2,linestyle=':',color='k')
                    ax[i][j].set_ylabel('output $Y_t$')
                    ax[i][j].set_yticks([Y_0,ss_new['Y'],ss_old['Y']])
                    ax[i][j].set_yticklabels(['$Y_0$','$Y^*_{new}$','$Y^*_{old}$'])
                    ax[i][j].set_ylim(ylim)
                elif fig_param['trajectories'][i*fig_param['subplots'][1]+j] == 'I':
                    # investment path
                    ylim = [min(I_path)-0.1*(max(I_path)-min(I_path)),max(I_path)+0.1*(max(I_path)-min(I_path))]
                    ax[i][j].plot(T_path[0:-Tmin],I_path[0:-Tmin], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[1],label='consumption $C_t$')
                    ax[i][j].plot(T_path[-Tmin-1:-Tmin+1],I_path[-Tmin-1:-Tmin+1], linewidth=1,marker='',linestyle=':',alpha=1,color=ec.clist_vibrant[1])
                    ax[i][j].plot(T_path[-Tmin:Tmax-Tmin+1],I_path[-Tmin:Tmax-Tmin+1], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[1])
                    ax[i][j].plot([0,0],ylim,linewidth=2,linestyle=':',color='k')
                    ax[i][j].set_ylabel('investment $I_t$')
                    ax[i][j].set_yticks([I_0,ss_new['I'],ss_old['I']])
                    ax[i][j].set_yticklabels(['$I_0$','$I^*_{new}$','$I^*_{old}$'])
                    ax[i][j].set_ylim(ylim)
                elif fig_param['trajectories'][i*fig_param['subplots'][1]+j] == 'C':
                    # consumption path
                    ylim = [min(C_path)-0.1*(max(C_path)-min(C_path)),max(C_path)+0.1*(max(C_path)-min(C_path))]
                    ax[i][j].plot(T_path[0:-Tmin],C_path[0:-Tmin], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[3],label='consumption $C_t$')
                    ax[i][j].plot(T_path[-Tmin-1:-Tmin+1],C_path[-Tmin-1:-Tmin+1], linewidth=1,marker='',linestyle=':',alpha=1,color=ec.clist_vibrant[3])
                    ax[i][j].plot(T_path[-Tmin:Tmax-Tmin+1],C_path[-Tmin:Tmax-Tmin+1], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[3])
                    ax[i][j].plot([0,0],ylim,linewidth=2,linestyle=':',color='k')
                    ax[i][j].set_ylabel('consumption $C_t$')
                    ax[i][j].set_yticks([C_0,ss_new['C'],ss_old['C']])
                    ax[i][j].set_yticklabels(['$C_0$','$C^*_{new}$','$C^*_{old}$'])
                    ax[i][j].set_ylim(ylim)
                elif fig_param['trajectories'][i*fig_param['subplots'][1]+j] == 'r':
                    # rental rate path
                    ylim = [min(r_path)-0.1*(max(r_path)-min(r_path)),max(r_path)+0.1*(max(r_path)-min(r_path))]
                    ax[i][j].plot(T_path[0:-Tmin],r_path[0:-Tmin], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[2],label='consumption $C_t$')
                    ax[i][j].plot(T_path[-Tmin-1:-Tmin+1],r_path[-Tmin-1:-Tmin+1], linewidth=1,marker='',linestyle=':',alpha=1,color=ec.clist_vibrant[2])
                    ax[i][j].plot(T_path[-Tmin:Tmax-Tmin+1],r_path[-Tmin:Tmax-Tmin+1], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[2])
                    ax[i][j].plot([0,0],ylim,linewidth=2,linestyle=':',color='k')
                    ax[i][j].set_ylabel('rental rate $r_t$')
                    ax[i][j].set_yticks([r_0,ss_new['r'],ss_old['r']])
                    ax[i][j].set_yticklabels(['$r_0$','$r^*_{new}$','$r^*_{old}$'])
                    ax[i][j].set_ylim(ylim)
                elif fig_param['trajectories'][i*fig_param['subplots'][1]+j] == 'w':
                    # wage path
                    ylim = [min(w_path)-0.1*(max(w_path)-min(w_path)),max(w_path)+0.1*(max(w_path)-min(w_path))]
                    ax[i][j].plot(T_path[0:-Tmin],w_path[0:-Tmin], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[5],label='consumption $C_t$')
                    ax[i][j].plot(T_path[-Tmin-1:-Tmin+1],w_path[-Tmin-1:-Tmin+1], linewidth=1,marker='',linestyle=':',alpha=1,color=ec.clist_vibrant[5])
                    ax[i][j].plot(T_path[-Tmin:Tmax-Tmin+1],w_path[-Tmin:Tmax-Tmin+1], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[5])
                    ax[i][j].plot([0,0],ylim,linewidth=2,linestyle=':',color='k')
                    ax[i][j].set_ylabel('wage $w_t$')
                    ax[i][j].set_yticks([w_0,ss_new['w'],ss_old['w']])
                    ax[i][j].set_yticklabels(['$w_0$','$w^*_{new}$','$w^*_{old}$'])
                    ax[i][j].set_ylim(ylim)
                elif fig_param['trajectories'][i*fig_param['subplots'][1]+j] == 'k':
                    # capital per capita path
                    ylim = [min(k_path)-0.1*(max(k_path)-min(k_path)),max(k_path)+0.1*(max(k_path)-min(k_path))]
                    ax[i][j].plot(T_path[0:-Tmin],k_path[0:-Tmin], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[6],label='capital $K_t$')
                    ax[i][j].plot(T_path[-Tmin-1:-Tmin+1],k_path[-Tmin-1:-Tmin+1], linewidth=1,marker='',linestyle=':',alpha=1,color=ec.clist_vibrant[6])
                    ax[i][j].plot(T_path[-Tmin:Tmax-Tmin+1],k_path[-Tmin:Tmax-Tmin+1], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[6])
                    ax[i][j].plot([0,0],ylim,linewidth=2,linestyle=':',color='k')
                    ax[i][j].set_ylabel('capital per capita $k_t$')
                    ax[i][j].set_yticks([k_0,ss_new['k'],ss_old['k']])
                    ax[i][j].set_yticklabels(['$k_0$','$k^*_{new}$','$k^*_{old}$'])
                    ax[i][j].set_ylim(ylim)
                elif fig_param['trajectories'][i*fig_param['subplots'][1]+j] == 'y':
                    # output per capita path
                    ylim = [min(y_path)-0.1*(max(y_path)-min(y_path)),max(y_path)+0.1*(max(y_path)-min(y_path))]
                    ax[i][j].plot(T_path[0:-Tmin],y_path[0:-Tmin], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[7],label='output $Y_t$')
                    ax[i][j].plot(T_path[-Tmin-1:-Tmin+1],y_path[-Tmin-1:-Tmin+1], linewidth=1,marker='',linestyle=':',alpha=1,color=ec.clist_vibrant[7])
                    ax[i][j].plot(T_path[-Tmin:Tmax-Tmin+1],y_path[-Tmin:Tmax-Tmin+1], linewidth=3,marker='',linestyle='-',alpha=1,color=ec.clist_vibrant[7])
                    ax[i][j].plot([0,0],ylim,linewidth=2,linestyle=':',color='k')
                    ax[i][j].set_ylabel('output per capita $y_t$')
                    ax[i][j].set_yticks([y_0,ss_new['y'],ss_old['y']])
                    ax[i][j].set_yticklabels(['$y_0$','$y^*_{new}$','$y^*_{old}$'])
                    ax[i][j].set_ylim(ylim)
    else:
        fig = None
        ax = None

    data = dict()
    data['fig'] = fig
    data['ax'] = ax
    data['T_path'] = T_path
    data['K_path'] = K_path
    data['Y_path'] = Y_path
    data['I_path'] = I_path
    data['C_path'] = C_path
    data['r_path'] = r_path
    data['w_path'] = w_path
    data['k_path'] = k_path
    data['y_path'] = y_path
    data['Tmin'] = Tmin
    data['p_old'] = p_old
    data['p_new'] = p_new
    data['ss_old'] = ss_old
    data['ss_new'] = ss_new

    return data


