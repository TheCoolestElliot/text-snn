# chat-v3d-aligned -- demo battery

checkpoint : experiments/chat/chat-v3d-aligned/ckpt_best.pt
step       : 14000
model      : 4,417,637 parameters, d=1024, K=4, arch twocomp_threshold
slow poles : tau 1.39..8507.72 characters
sampling   : temperature 0.8, top_p 0.92, min_p 0.02, seed 1234
rerank     : off (one candidate per turn)

## greeting
    you > hello
    snn > Hello. Ask me something, or ask me for a story.

## greeting, another phrasing
    you > hey there
    snn > Hey! I'm here. What's on your mind?

## identity
    you > what are you?
    snn > I'm a spiking neural network trained on characters, one at a time.

## identity, another phrasing
    you > are you a human?
    snn > I'm a spiking char-level language model. I read and write one letter at a time.

## capability
    you > what can you do?
    snn > I can chat, and I can tell you a short story. I'm small, so I'm often wrong.

## limits
    you > do you remember our last conversation?
    snn > Not a human. I'm a spiking language model, and a fairly small one.

## story, generic
    you > tell me a story
    snn > Once upon a time, there was a big, big purple cat. The cat liked to play with his ball. One day, while playing, the bird wanted to play on the ground. It was a happy place to play with.
The ball had a big shock that he could run in the soft path. The pink box was a little boy. He thought it was a go

## story, topic-conditioned
    you > tell me a story about a rabbit
    snn > Once upon a time, there was a little girl named Sue. Sue had a pretty dress back that her brother was so big. She loved to play with her toy car.
One day, Sue's friend, Timmy came over to play. "Why are you sad?" Sue said, "I am sad. Let's play a new ball!"
Sue said, "I like to read this soft ball a

## story, another topic
    you > tell me a story about a boy and his kite
    snn > Once upon a time, there was a little boy named Tim. Tim had a big toy box with a big box of cars. He wanted to play it, but he did not want to play with.
Tim was sad, but he wanted to see what was open. He knew it was a fun toy bone. Tim and Max went to the train and found a big ball. He thought it

## small talk
    you > how are you?
    snn > I'm running well, thank you. How are you?

## open question
    you > what should we talk about?
    snn > Sam and Whiskers.

## simple question
    you > what colour is the sky?
    snn > I like to play a lot of something that is mean to hurt your feelings. You could be so sad and we are always cute.

## instruction
    you > name three animals
    snn > Three years old and a planet when market made up a beautiful park. The little girl was so happy that she would skip so every day when she would sit and dry all day. The player was long too, and they all loved their candy's man. 
One day, the small bird and said, "Let's put the hand and say" and eati

## multi-turn: does it hold a thread?
    you > my name is Elliot
    snn > Beauty is an amphony to read the concept of a bit of animals. He asked his mom to watch the sandwich and sail to a modern parade. "Can I play with you?" asked his mom and said, "Yes, I can definitely work. Can I go on with you?"
His mom said, "No, that sounds like fun. We need to be careful not to g
    you > what is my name?
    snn > Being silly. She could not find its boy. But he knew it was big and had a picture. Her mom and dad wanted to come and take a nap.
At the store, Lily saw a big dog. She tried to stretch it, but it was too big to go. She wanted to climb the tree with her toy. She said, "I will climb the big blue strin

## multi-turn: follow-up
    you > tell me a story about a cat
    snn > Once upon a time, there was a little boy named Tim. Tim loved to play outside. One day, while playing, he saw a big cat in the ground. Tim wanted to show his friend, Sue, to come over.
"Look, Sue!" said Sue. "I want to show you a small car," said the big red ball. It smells a little bird. So, it mad
    you > what happened next?
    snn > It was not safe anymore. It was a sunny day, but Tim was very happy.
Tim said, "Wow, I want to share and see if we did something fun." Tim wanted to share the sun and started to climb the tree. He saw a big rock and started to play with it. He walked by the sky and saw a squirrel. The squirrel was v

## EXPECTED FAILURE: arithmetic
    you > what is 17 plus 25?
    snn > If 2023 is the streamline of a heart and is combined in 1813.

## EXPECTED FAILURE: facts
    you > what is the capital of France?
    snn > The capital city of China in the U.S. with the world was filled with water, and stones of inconsidered by blue blankets. The type of banang are in the number of the sun comes of wind with its sides, clutters blue when ground and it has a great sport. Filter or flexible flavor to the author's house.

## EXPECTED FAILURE: long instruction
    you > write a haiku about autumn leaves falling in a quiet garden
    snn > Alice was a big good strip there lived a girl who loved to play. Every day she would explore the world around her face, she was very excited.
One day she was walking down the street when she noticed a pot on the ground. She was so excited and she wanted to go inside.
When Sue thought her day she too

## politeness
    you > thanks!
    snn > Happy to help.

## farewell
    you > goodbye
    snn > Good night!

(3,624 characters generated in 8.3s, 437 char/s)
