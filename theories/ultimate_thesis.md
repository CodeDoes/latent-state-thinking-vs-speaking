# Ultimate Thesis — Verbatim

> **The core thesis** of `latent-state-thinking-vs-speaking`.  
> **Date range**: 2026-07-10 to 2026-07-17

---

## The Core Verbatim

"the ultimate goal is to produce something novel by doing small experiments. my 'ultimate theory' is that machine learning can be done with a smaller system. and we can prove 1 thing at a time instead of waiting for emergant properties"

"the core of the learning will be read-many-context-tokens answer with few but verifiable tokens"

"my ultimate goal is to make a latent that can induce many tokens. instead of processing the whole context repeatedly and outputting 1 token. then processing the whole context + 1 and outputting the next. i process the context and derive the future and after that rapidly decode the remaining tokens required."

"we are trying to move the latent into the correct orientation. we are trying to derive an answer from input. my 'GRAND' idea was to combine transformer and state model. keep the context small so the model only needs to handle small parts at a time."

"basically 'small models, fast experiments, prove one thing at a time'"

"the ultimate goal is to produce something novel. small proof. don't wait for emergant properties."

"i'd rather start from the underlying assumptions. (though my assumptions are based on hearing from other people's previous research)"

---

## Sub-Themes (Verbatim Collapsed)

### State-based architectures

"process bytes quickly until you are surprised, create a patch from the bytes, pass that patch to the inner, inner passes a new state to a decoder, decoder creates logits?"

"byte -> surprised? go to patch-model with byte-level-state as patch : get new byte-level-state and repeat ingest"

"encoder-decoder are trained on surprise at the next byte. and the patch should be trained on surprise at the next patch. the assumed result would be a model that can have longer range prediction than a simple byte level model"

"i think you do not need to look at each layer. you can just look at the resulting state. there are ephemaral states (that basically just hold the recent tokens and then long running states. that remain for longer."

"if we could be sure that the state-vector was moving from start to finish we could do it at the boundry."

"theoretically for SSM to extract the location it only needs to trace index and current_location_at_index"

"the state is an array of vectors"

### Multi-rate / hierarchical

"i think fundamentally the byte-state and patch-state should be seperate. (byte-state+patch-state) -> patch-model and also into byte-model"

"byte space RWKV -> latent space RWKV -> byte space RWKV"

"BLRNN = makeloop(backprop(RWKV layer * max_loop_const) extra=earlyExitTrigger) MODEL = BLRNN + PATCHRNN + BLRNN"

"Ingest ...[state,byte]-> encoder-model [state] -> encoder-model | [state,patch] -> patch-model [state,patch]-> decoder-model [state,byte]-> ...generate"

### Looping / adaptive

"you always train the looping model to the full max_loop_count. but you train it to predict how far from complete it is given the context"

"BLRNN + PATCHRNN + BLRNN" — basic three-component looped design

"SSM trains towards confidence and FFN trains towards complete. (latent-state+context -> SSM -> repeat on confidence<1.0 or (latent-state+context -> FFN -> repeat on context_complete<1.0))"

"my idea of why the patch model fails to train: it needs to learn byte-level-state and also needs to learn patterns. the research i did proves that you can safely increase the patch dimensions without being excessive."

### Anatomy of the model

"# encoder-model
input: encoder-state+byte
output: encoder-state+trigger

# decoder-model
input: encoder-state+decoder-state+patch
output: decoder-state+trigger

# patch-model
input: model-state+patch
output: model-state+patch"

"(proto-patch+byte) -> encoder -> (proto-patch)
proto-patch == patch? patch = proto-patch : repeat
patch -> processor -> decoder
patch -> decoder
decoder -> (patch+byte)"

"this is the most logical way i can think of. byte -> encoder -> patch. is patch surprising? if not read next token."

### Mixed architectures

"a small pure BLT model -> working
a small BLT encoder and decoder with a small RWKV -> working
a RWKV based encoder and decoder with a normal transformer -> working
a pure RNN based BLT model -> working

in that order. would be better"

### Future goals

"if we could be sure that the state-vector was moving from start to finish we could do it at the boundry. and we could like have 2 states (new/old) and we just check when old dissapears?"

"we can think of it as having 3 input and output (old-state-vector, new-state-vector, byte [256 onehot?]) and the output (new-state-vector, swap-trigger)"

"i would be more interested in a AI that can listen to Wifi and Bluetooth and all the different devices and streams on the pc and find information from that."

"can't we rather consider things we might use the training time for. ... is there not something we can deliberately overfit it on."

### Failures (Verbatim Diagnoses)

"i think this could have exited almost immediately. looking at the loss of all. there was NO movement. only the autoencoder moved. meaning you are doing something wrong"

"i think i have a theory of why. i think the encoder-decoder does not have a large enough state. try focusing on the encoder decoder for now."

"my idea of why the patch model fails to train: it needs to learn byte-level-state and also needs to learn patterns. ... but first i think we need to make the patch model 3 times larger than the encoder+decoder."

---

## Final Synthesis (User-Verbatim Sentiment)

"i want something where i can use ai in realtime and that constantly learns"

"i don't want only dendrite meta-info. basically i want you to restore my verbatim."

"im brain storming and poking the ai to explore its latent space"
