import torch
def load_ckpt_from_state_dict(path,model):
    try:
        state_dict = torch.load(path)
        if "module." in list(state_dict.keys())[0]:
            print("Tag 'module.' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("module.", "")] = state_dict.pop(key)
        if "swin_vit" in list(state_dict.keys())[0]:
            print("Tag 'swin_vit' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
        
        model.load_state_dict(state_dict, strict=False)
        print("Using pretrained self-supervised Swin UNETR backbone weights !")
        return model
    except ValueError:
        raise ValueError("Self-supervised pre-trained weights not available ")