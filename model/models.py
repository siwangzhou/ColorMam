
def create_model(opt):
    from .colorhistogram_model import ColorHistogram_Model
    model = ColorHistogram_Model()        
    model.initialize(opt)
    return model

def create_ex_model(opt):
    from .same_ex_model import same_ex_model
    model = same_ex_model().cuda()
    model.initialize(opt)
    return model
