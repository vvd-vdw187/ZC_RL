import torch 
import torch.nn as nn

#https://github.dev/SamsungLabs/zero-cost-nas
def GraSP(model, input, targets, mode, loss_fn, iters=1):
    # targets not feasable?

    # get all weights from the model
    weights = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)

    model.zero_grad()
    N = input.size(0)
    #for 
    # split data?
    #TODO do this with batches.
    grad_w = None
    for _ in range(iters):
        output = model.inference(input)
        # Perform a forward pass in the environment
        loss = model.loss(output)

        grad_w_p = torch.autograd.grad(loss, weights, allow_unused=True)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for i in range(len(grad_w)):
                grad_w[i] += grad_w_p[i]

        



        # # calculate loss
        # loss = loss_fn(output, targets)
        # # backward pass
        # loss.backward()
        # # calculate gradient
        # grad_w = []
        # for layer in model.modules():
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         grad_w.append(layer.weight.grad)


    
