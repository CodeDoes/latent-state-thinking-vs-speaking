# Working Method — Verbatim

> **Date range**: 2026-07-10 to 2026-07-17  
> **You said**:

"okay. take user and write PROGRESS.md"

"read @USER_base.md and update PROGRESS and make a PLAN.md"

"git commit. also do you understand that I want you to use the kaggle api?"

"update AGENTS.md make sure it explains that I WANT YOU TO USE KAGGLE FOR EVERY EXPERIMENT!"

"continue with this. git commit. and try to use resuable scripts"

"write what to use and maintain in AGENTS.md"

"okay can we keep on experimenting. i liked the previous experiment but i feel it gave no feed back. i couldn't see the result over time. so i can't infer the causes and what mechanism would be better or if there was a lack of data diversity or the training or math was wrong."

"update @AGENTS.md so you do not try to train on local machine"

"try again. are we making progress vs how long would similar code take on my pc."

"i have gpu ../roco_ai/ ../rwkv-harness/ devenv.nix"

"i think i stopped using kaggle for now. im not training massive models"

"i want you to write theories/*.md and keep track of experiments on theories in PROGRESS.md"

"git commit. how we going to experiment? can you start?"

"write a research-paper with theory, experiment description and results so i can evaluate this later"

"do not ask questions which can not be inferred. ask questions with only 1 answer. like i'd understand if it said 'went to X and gave Y'"

"git commit. why you doing it like this. make a script that we can reference later"

"theories/*.md and theories/*.infer.md

the normal md is mine the infer is your interpretation of my theory. only include details not explained in my md .

Write this to AGENTS.md"

"update AGENTS.md im no longer using kaggle. only local.
remove README.md and requirements.txt. use devenv instead."

"add to AGENTS.md that when i make a message you can edit theories md file with my verbatim words"

"i want the training to be observable. i prefer to have resumable training. i prefer to constantly improve the trained model. experiments should attach their code to a git commit hash. trained models should be easy to clear.

my plan is to train RWKV nano to simulate a real program.

you need to write a synthdata-generator and then a synthdata-solver.

we need to somehow train its logic. you can train it on procedurally generated content.

any thoughts?"

"either fix your error. or if the experiment proves the theory is invalid (must be able to prove this from looking at the loss and reasoning about the training data) then you can create a different theory that might unlock a different advantage instead. keep track of experiments and git commit before launching another round. continue with your self-directed research."

"go on. git commit for each step. you are doing well."

"add experiment. progressive expansion. basically you train a model on a task and then if you want to add some sort of extra capbility to it you run a sample of the new-task through it and monitor the activations for any sort of bottleneck (gate) and you use that as a marker for a good place to add additional layers or concentrate learning relative to.
theory is that you can do this in a way that would not be obstructive. and also can naturally expand the model. and also that the training for th"

"i'd rather start from the underlying assumptions. (though my assumptions are based on hearing from other people's previous research)"

"yes. i did not want to change everything at once. i wanted multiple hypothesis and experiments. obviously isolate things that can be issolated. and those that can't try to introduce them sequentially."

"OKAY git commit. they start the next."

"commit all. let's chat. first i want to do the training with a BLT variant of RWKV so i can fix the technical glitches that will inevitably come up. and only after we have a solid foundation do i want to do a training run (GPU) and prove/disprove theories."

"i think fundamentally the byte-state and patch-state should be seperate. (byte-state+patch-state) -> patch-model and also into byte-model

the encoder might not need the patch-state. but the decoder will. so i think its better if we just use the same architecture for both."

"i'm thinking of stepping away from the pure RNN. this is getting a bit ahead of me. a small pure BLT model -> working
a small BLT encoder and decoder with a small RWKV -> working
a RWKV based encoder and decoder with a normal transformer -> working
a pure RNN based BLT model -> working

in that order. would be better"

"start. git commit after each experiment is recorded"

"theories can get simplified and archived in git history instead. can also record a snapshot of experiment results somehow 'X proves Y' for later reference"

"theories/ is for my crazy verbatim utterings!
theories/*.infer is what you made sense of
theories/*.state is how the experiements are going
experiments/ are for experiments for all the underlying assumptions and the hypothesis and claims in theories/
reports/ is for outside consumption on what has been discovered so far."

"i don't want only dendrite meta-info.
basically i want you to restore my verbatim. and recreate the infer and clean up the status inside theories/"

"you should extract even the trash ! you think i don't know im messy. its not like it matters. im not talking to a person. im brain storming and poking the ai to explore its latent space."
