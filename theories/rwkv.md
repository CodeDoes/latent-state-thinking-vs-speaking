make a small rwkv model and we are going to find out how quickly it can train and how strong we can make it for *one specific capability* (something easy to validate). not general capabilities.

if we train quickly enough we can fix the synthdata generator until overfitting is no longer possible.

my plan is to train RWKV nano to simulate a real program.

we need a synthdata-generator and a synthdata-solver.

we need to somehow train its logic. you can train it on procedurally generated content.

the core of the learning will be: read-many-context-tokens, answer with few but verifiable tokens.

we can also train a logic niiah. instruction+noise+needle+noise+needle-action-transformation+noise+repeat-X-times+ask-questions-about-needle-transformations

i want the training to be observable. i prefer to have resumable training. i prefer to constantly improve the trained model.

experiments should attach their code to a git commit hash. trained models should be easy to clear.
