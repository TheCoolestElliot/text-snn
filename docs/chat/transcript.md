# chat-v3d-aligned -- demo battery

checkpoint : experiments/chat/chat-v3d-aligned/ckpt_best.pt
step       : 14000
model      : 4,417,637 parameters, d=1024, K=4, arch twocomp_threshold
slow poles : tau 1.39..8507.72 characters
sampling   : temperature 0.8, top_p 0.92, min_p 0.02, seed 1234
rerank     : n=8 lambda=0.6 min_chars=12 null=empty_user

## greeting
    you > hello
    snn > Hello! I'm listening.

## greeting, another phrasing
    you > hey there
    snn > Hey! I'm here. What's on your mind?

## identity
    you > what are you?
    snn > I'm a spiking char-level language model. I read and write one letter at a time.

## identity, another phrasing
    you > are you a human?
    snn > Not a human. I'm a spiking language model, and a fairly small one.

## capability
    you > what can you do?
    snn > Not much, honestly. I chat, and I tell short stories.

## limits
    you > do you remember our last conversation?
    snn > I can't. I have no connection to anything outside this computer.

## story, generic
    you > tell me a story
    snn > Once upon a time, there was a little girl named Lily. She loved to play with her toys and explore the world. One day, she went for a walk with her mom. So, she was very excited and wanted to take a big pile of paper.
Mia asked her mom, "Can we race the big house for your favorite clothes?"

## story, topic-conditioned
    you > tell me a story about a rabbit
    snn > Once upon a time, there was a little dog named Max. Max loved to play with his friend, a little bird named Sam. They loved to play and have fun together.
One day, Mimi saw a big, red ball outside. The cow was very exploring. Max wanted to eat the ball, but it was too high.

## story, another topic
    you > tell me a story about a boy and his kite
    snn > Once upon a time, there was a little boy named Tim. Tim loved to play and run around. One day, he went to the park with his mom. At the park, he saw a big tree. Tim thought it would be fun to play with his friends.
At the park, Tim met a new friend named Sue. Sue saw the pumpkin and walked.

## small talk
    you > how are you?
    snn > All my neurons are firing. How are you?

## open question
    you > what should we talk about?
    snn > Sam and Whiskers.

## simple question
    you > what colour is the sky?
    snn > I love to read a book.

## instruction
    you > name three animals
    snn > That's not something I can have a view on.

## multi-turn: does it hold a thread?
    you > my name is Elliot
    snn > Shipper was a little girl named Lily who loved to play with her toys. One day, Lily was playing with her toy tea. She put on her toy to play on the swings and have fun. Lily was very happy.
Lily went to the park with her mom. She looked at the sky and saw her a big wing. "Why are you so sad?"
    you > what is my name?
    snn > Mostly an egg sticker. It was soft and fluffy with others and tapes. She liked the wind blow and splash in the park. 
The little girl and the little girl went to play outside. The sun was shining and the birds were singing. The bird was very happy. She asked her friends, "Look at my belonging!"

## multi-turn: follow-up
    you > tell me a story about a cat
    snn > Once upon a time, there was a little girl named Lucy. She had a pretty spoon in her room. She wanted to go up the house and picked it up with her toys.
Lily took the soup and started to complete the doll. It was very old and a fun game with many friends. They played together and had fun together.
    you > what happened next?
    snn > I'm good to see if they could go and get started.

## EXPECTED FAILURE: arithmetic
    you > what is 17 plus 25?
    snn > The total depends on what to do.

## EXPECTED FAILURE: facts
    you > what is the capital of France?
    snn > The thermost tree in the law area of the car in the United States is convented in the control of the Earth.

## EXPECTED FAILURE: long instruction
    you > write a haiku about autumn leaves falling in a quiet garden
    snn > Sure! I can't believe it! What have you been put for this for you?

## politeness
    you > thanks!
    snn > Happy to help.

## farewell
    you > goodbye
    snn > See you. Have a good one.

(2,470 characters generated in 14.6s, 169 char/s)
