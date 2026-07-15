# progressive-expansion

add experiment. progressive expansion. basically you train a model on a task and then if you want to add some sort of extra capbility to it you run a sample of the new-task through it and monitor the activations for any sort of bottleneck (gate) and you use that as a marker for a good place to add additional layers or concentrate learning relative to.

theory is that you can do this in a way that would not be obstructive. and also can naturally expand the model. and also that the training for the aditional capability will be faster if just concentrated on the layers relative to the gate.

im not sure about everything but i think its possible.
