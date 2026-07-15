# generation-loss

i think we should allow the model to generate after reading context and record each generation's logits and apply backprop on the loss for each wrong token (against the correct answer)
