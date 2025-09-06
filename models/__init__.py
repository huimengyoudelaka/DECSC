from models.CSCNet_T_grad_DEQ import CSCNet_T_grad_DEQ

def cscnet_t_grad_deq(params):
    net = CSCNet_T_grad_DEQ(params)
    net.use_2dconv = True
    net.bandwise = False
    return net
