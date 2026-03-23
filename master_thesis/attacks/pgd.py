import torch
import torch.nn as nn

def pgd_attack(model,images,labels,eps,alpha,iters,device):
    '''
    model: the model to attack
    x: input image
    label: true label of the input image
    eps: maximum perturbation
    alpha: step size
    iters: number of iterations
    
    '''
    original_images= original_images.clone().detahc().to(device)
    adversarial_images= original_images.clone().detach().to(device)

    for i in range (iters):
        adversarial_images.requires_grad=True
        outputs=model(adversarial_images)
        loss= nn.CrossEntropyLoss()(outputs,labels)
        
        model.zero_grad() #reset gradients to zero
        loss.backward()
        # disabling gradient calculation 
        with torch.no_grad():
            gradient=adversarial_images.grad.data
            adversarial_images=adversarial_images+alpha* gradient.sign()

            perturbations= torch.clamp(adversarial_images-original_images, min=-eps, max=eps)

            adversarial_images= torch.clamp(original_images+perturbations, min=0, max=1).detach()
        
        adversarial_images=adversarial_images.detach()

    return adversarial_images,perturbations
