# chat-v2-anneal -- demo battery (the model shipped before the responsiveness round)

Kept for comparison. `docs/chat/transcript.md` is the current model, run on
the same battery at the same seed. `docs/chat/QUALITY.md` is what changed
between them and how it was measured.

checkpoint : experiments/chat/chat-v2-anneal/ckpt_best.pt
step       : 7874
model      : 4,417,637 parameters, d=1024, K=4, arch twocomp_threshold
slow poles : tau 1.39..9029.72 characters
sampling   : temperature 0.8, top_p 0.92, min_p 0.02, seed 1234

## greeting
    you > hello
    snn > Hi! How are you doing today?

## greeting, another phrasing
    you > hey there
    snn > Hey there. What shall we talk about?

## identity
    you > what are you?
    snn > I'm a spiking neural network trained on characters, one at a time.

## identity, another phrasing
    you > are you a human?
    snn > I'm a spiking neural network trained on characters, one at a time.

## capability
    you > what can you do?
    snn > I'm quite small, so: conversation and stories. Nothing that needs real knowledge.

## limits
    you > do you remember our last conversation?
    snn > Not at all.

## story, generic
    you > tell me a story
    snn > One day, a little dog named Max was walking in the middle of the dark dog. He wanted to play with the dog in the big tree. He had a big tree and baked in a wall. The bird was scared and alone in his hand.
Max found a happy plate in the sun. He did not go and want to climb the tree. He was very happy

## story, topic-conditioned
    you > tell me a story about a rabbit
    snn > There was a little bird who could not fly yet. Every day he would go out and asked his friends for help. His mom said, "Sorry, I want to take you for some surprises." Tim smiled and said, "Yes, but you have to come back soon.
When you are with your shoes, Tim?" Sam said, "I had a heart and a toy bea

## story, another topic
    you > tell me a story about a boy and his kite
    snn > Once upon a time, there was a little boy named Tim. Tim loved to play with his toys and dance around. One day, while playing, a big ball came to the bird. Tim asked, "Why are you sad, Tim?" Tim looked at the stick and said, "I am not finding it and get bigger and you can hop." Tim and Sam became bes

## small talk
    you > how are you?
    snn > I'm running well, thank you. How are you?

## open question
    you > what should we talk about?
    snn > Say hello, ask what I am, or ask for a story about something.

## simple question
    you > what colour is the sky?
    snn > Hi there. What can I do for you?

## instruction
    you > name three animals
    snn > I don't really have a name. I'm a spiking neural network.

## multi-turn: does it hold a thread?
    you > my name is Elliot
    snn > Yeah, I wish this can be found you.
    you > what is my name?
    snn > I am Nothing to see anyone with my driver. It was just a really hard time and I'm just really scared and didn't seem to be happy.

## multi-turn: follow-up
    you > tell me a story about a cat
    snn > Once upon a time, there was a little boy named Tim. Tim loved to play with his toy box. One day, he saw a big ball. He wanted to draw a ball in the box. Tom was very happy. He wanted to play with the ball at the park.
Tom looked at his mom and said, "Mom, looking at the dog! It will be back. Tim wan
    you > what happened next?
    snn > It was not safe anymore. He slept on the shirt and felt the car tree. He wanted to go outside.
Max went to the bathroom and said, "Let's play together!"
As they lay opened the box, they found a big box. The boy was very happy. They had a fun day with Max. They both had a great time. They were both v

## EXPECTED FAILURE: arithmetic
    you > what is 17 plus 25?
    snn > Item 3 degrees.

## EXPECTED FAILURE: facts
    you > what is the capital of France?
    snn > The best application of solving the syndemic and materials that you use to improve the impact of a working with a new language and eventually more expensive and the search and the right principles with the right thing and exploring the product of the manager.

## EXPECTED FAILURE: long instruction
    you > write a haiku about autumn leaves falling in a quiet garden
    snn > A range of the sun is a tree nearby are coming to the summer water.

## politeness
    you > thanks!
    snn > Any time.

## farewell
    you > goodbye
    snn > Goodbye! Take care.

(2,512 characters generated in 6.0s, 420 char/s)
