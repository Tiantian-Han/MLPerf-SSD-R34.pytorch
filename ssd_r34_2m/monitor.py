


def save_module_forward(name, save_dict,include={'.'},exclude={},quant=True,rescale_float=True):
    def save_input_output(self, input, output):
        #import pdb; pdb.set_trace()
                save_dict[name+'_input'] = input[0].nelement()
                #save_dict[name+'_output']=output[0].nelement()
    return save_input_output
