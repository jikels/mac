class hook():
    ''' 
    A simple hook class that returns the input and output of a layer during forward/backward pass
    Source: https://www.kaggle.com/sironghuang/understanding-pytorch-hooks
    
    self.online_model._modules.items()[module of list_of_modules][item_list of module_item_list][item of item_list]

    Module Hook Examples: 
    E.g. first layer: [Hook(list(self.online_model._modules.items())[1][1][0])] / [list_of_modules -> 1 layers][list_of_layers -> 1 layerlist][first layer]
    
    hook_first_layer = [Hook(list(self.online_model._modules.items())[1][1][0])]
    Inputs -> hookF[0].input
    Outputs -> hookF[0].output

    E.g. embedding of embeddings: [Hook(list(self.online_model._modules.items())[2][1])] / #[list_of_modules -> embeddings][list_of_embeddings -> embedding]
    
    hook_embedding = [Hook(list(self.online_model._modules.items())[2][1])]
    Inputs -> hookF[0].input
    Outputs -> hookF[0].output
    '''

    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()