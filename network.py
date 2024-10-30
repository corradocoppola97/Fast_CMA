import torch
import torchvision

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

def get_pretrained_net(type_model, num_classes, pretrained=True):

    if type_model=='resnet18':
        if pretrained:
            pretrainedmodel = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.resnet18()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel)

    elif type_model=='resnet50':
        if pretrained:
            pretrainedmodel = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.resnet50()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel)

    elif type_model=='resnet152':
        if pretrained:
            pretrainedmodel = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.resnet152()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel)
            
    elif type_model=='mobilenet_v2':
        if pretrained:
            pretrainedmodel = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.classifier[1].in_features
            pretrainedmodel.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.classifier[1].weight)
        else:
            pretrainedmodel = torchvision.models.mobilenet_v2()
            num_ftrs = pretrainedmodel.classifier[1].in_features
            pretrainedmodel.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel)
        
    elif type_model=='wide_resnet50':
        if pretrained:
            pretrainedmodel = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.fc.weight)
        else:
            pretrainedmodel = torchvision.models.wide_resnet50_2()
            num_ftrs = pretrainedmodel.fc.in_features
            pretrainedmodel.fc = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel)
    
    elif type_model=='efficientnet_v2_l':
        if pretrained:
            pretrainedmodel = torchvision.models.efficientnet_v2_l(weights=torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.classifier[1].in_features
            pretrainedmodel.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.classifier[1].weight)
        else:
            pretrainedmodel = torchvision.models.efficientnet_v2_l()
            num_ftrs = pretrainedmodel.classifier[1].in_features
            pretrainedmodel.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel)
        
    elif type_model=='swin_t':
        if pretrained:
            pretrainedmodel = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.head.in_features
            pretrainedmodel.head = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.head.weight)
        else:
            pretrainedmodel = torchvision.models.swin_t()
            num_ftrs = pretrainedmodel.head.in_features
            pretrainedmodel.head = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel)
    
    elif type_model=='swin_v2_t':
        if pretrained:
            pretrainedmodel = torchvision.models.swin_v2_t(weights=torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.head.in_features
            pretrainedmodel.head = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.head.weight)
        else:
            pretrainedmodel = torchvision.models.swin_v2_t()
            num_ftrs = pretrainedmodel.head.in_features
            pretrainedmodel.head = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel)
        
    elif type_model=='swin_b':
        if pretrained:
            pretrainedmodel = torchvision.models.swin_b(weights=torchvision.models.Swin_B_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.head.in_features
            pretrainedmodel.head = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.head.weight)
        else:
            pretrainedmodel = torchvision.models.swin_b()
            num_ftrs = pretrainedmodel.head.in_features
            pretrainedmodel.head = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel)

    elif type_model=='swin_v2_b':
        if pretrained:
            pretrainedmodel = torchvision.models.swin_v2_b(weights=torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1, progress=True)
            num_ftrs = pretrainedmodel.head.in_features
            pretrainedmodel.head = torch.nn.Linear(num_ftrs, num_classes)
            torch.nn.init.xavier_uniform(pretrainedmodel.head.weight)
        else:
            pretrainedmodel = torchvision.models.swin_v2_b()
            num_ftrs = pretrainedmodel.head.in_features
            pretrainedmodel.head = torch.nn.Linear(num_ftrs, num_classes)
            initialize_weights(pretrainedmodel)
    
    elif type_model=='maxvit_t': # NO X CIFAR
        pretrainedmodel = torchvision.models.maxvit_t(weights=torchvision.models.MaxVit_T_Weights.IMAGENET1K_V1, progress=True)
        num_ftrs = pretrainedmodel.classifier[5].in_features
        pretrainedmodel.classifier[5] = torch.nn.Linear(num_ftrs, num_classes)
        torch.nn.init.xavier_uniform(pretrainedmodel.classifier[5].weight)
    
    return pretrainedmodel

