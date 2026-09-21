# Side by side: shipped checkpoint vs chat-v6-scratch, same prompts, same sampler seed, same decoder (n=256, lam=0.6)

Selection rule fixed BEFORE reading any reply: sampler seed 0; the first 6 prompts of HELDOUT and the first 4 of WIDE, in file order. Not cherry-picked. Replies are stored truncated to 300 chars. `[hit]` = names the requested noun.

## heldout: "tell me a story about a penguin"

**shipped (chat-v3d-aligned)** [miss]

> Once upon a time, there was a grumpy muffin. It was a close day outside. It wanted to go and talk to the people who wanted it. They were very excited. / The moral of the story to eat the fork and got a big storm came out of the tower. It was very pretty and said, "Maybe I can play with you, coach."

**chat-v6-scratch** [hit]

> Once upon a time, in a small house, there was a penguin. The pig loved to play with his friends. One day, he found a big piece of paper. He picked it up and started to put it on it. He was very happy. / The pine did not like the bug to perform a gift. Then, a gift wiped on his first weight.

## heldout: "tell me a story about a lighthouse"

**shipped (chat-v3d-aligned)** [miss]

> Once upon a time, in a big forest, there lived a little girl named Lily. She loved to help her friends with her friends. They liked to play and laugh together. One day, they decided to tell sing a sandwich. / Lily saw a big tree and wanted to climb it. But, Lily saw a big red ball in the ground.

**chat-v6-scratch** [miss]

> Once upon a time, there was a gray cat named Tom. Tom liked to wandering his toys all day long. One day, Tom saw a big red ball on the ground. He was very happy and wanted to play with the toys. / Tom said to his mom, "Mom, can you tell me and put a flower and said yes!"

## heldout: "tell me a story about a wizard"

**shipped (chat-v3d-aligned)** [miss]

> Once upon a time, there was a little bunny named Ben. Ben loved to eat fluffy very fast. He liked to tell his friends the other animals in the town. One day, Fred saw a big tree near the plane. / "Hi, red bird!" said Tim. "Do you want to play with me?" / Fin was scared.

**chat-v6-scratch** [miss]

> Once upon a time, there was a little girl named Lily. She had a big house with many colors. Lily loved to play with her toys and tell her mom about the way to make her toys. / One day, Lily's mom told her that they were on the kitchen. Lily had a big, colorful butterfly.

## heldout: "tell me a story about a turtle"

**shipped (chat-v3d-aligned)** [hit]

> One day, a little girl named Lily wanted to play with her toys in the park. She liked to run, jump, and get it with her toy drum. The turtle was very old and could not help. Lucy took the guitar and said, "I will play the mole with my bug!"

**chat-v6-scratch** [hit]

> Once upon a time, there was a little boy named Tim. Tim loved to play outside in the sun. One day, Tim went to the park to play with his friends. / Tim saw a big bug on the ground. He picked it up and saw a little girl named Lily. She was scared of Lily. They both liked the turtle.

## heldout: "tell me a story about a violin"

**shipped (chat-v3d-aligned)** [miss]

> One day, a little boy named Tim found a shiny ball in the grass. He thought it was a good idea for showing it because it was a fun place to play. He was very happy and said, "Hi, I want to do it!" / Tim said, "We will see the chain and tell story to get the path."

**chat-v6-scratch** [miss]

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and read about her shapes. One day, she went to her mom's room to the store. Lily was very happy to be friends. / As Lily started to unint, there was a special story. She was very excited to explore.

## heldout: "tell me a story about a castle"

**shipped (chat-v3d-aligned)** [hit]

> Once upon a time, there was a big castle. The car liked to play in the water. One day, the cat was playing with his ball. The cat wanted to play with the ball, but the cat was not his. / The cat said, "I want to help you!" The cat did not listen.

**chat-v6-scratch** [hit]

> Once upon a time, there was a little caterping. The cat was very pretty. The cat liked to share his friends until the kind cat would cook. The cat was a little boy who liked to play in the castle. / One sunny day, the cat found a big rock.

## wide: "tell me a story about a palace"

**shipped (chat-v3d-aligned)** [miss]

> Once upon a time, there was a boy named Tom. He loved to eat polite. One day, he found a big, thick cat named Tom. Tom said, "Hello, Mr. Snop! I like to tell you what I want!" / Tom was very cold. He tried and tried, but he wanted to go to the distance stories.

**chat-v6-scratch** [hit]

> Once upon a time, there was a big party. In the palace, there was a kind turkey. The palm was very frustrated. The toy was full of green stars. / One day, a little boy named Tim saw a grumpy old man. Tim was a big red ball when he put it. He asked his mom, "Can I play with you?" / Tim said, "Yes, Tim.

## wide: "tell me a story about a reindeer"

**shipped (chat-v3d-aligned)** [miss]

> Once upon a time, there was a big race with whole flowers. He lived in a big cave in the woods. He was a happy dog who loved to play and laugh. One day, he saw a big tree and wanted to catch it. / The cat said, "I can tell you to help me with this mouse. Can you help me?"

**chat-v6-scratch** [miss]

> Once upon a time, there was a red bird who lived in a big tree. The bird loved to play and run all day. The tree had a fancy store. One day, the bird saw a small bird in a tree. / The bird wanted to tell his friend, the dog. The bird went to her. The bird asked the tree and said, "Thank you, bird!

## wide: "tell me a story about a billboard"

**shipped (chat-v3d-aligned)** [miss]

> Once upon a time, in a big forest, there lived a little girl named Amy. She loved to play with her toys and run around the sun. One day, she found a big box of socks. She sat on the box and found a story in the pot. / Amy said, "Hi big brother, Tim! You should have to show it to see the toy there.

**chat-v6-scratch** [hit]

> Once upon a time, there was a big billboard in the sky. The bird had a pink bird, blue, and she loved to play and run. They had a lot of fun together. / One day, the bird saw a little bird. The bird said to the bird, "Why are you sad?" asked the bird.

## wide: "tell me a story about a buckle"

**shipped (chat-v3d-aligned)** [miss]

> Once upon a time, there was a little boy named Tim. Tim loved to wear plants. He would tell his friends the crocodile noticed that he was eager to find something that was very old. / One day, Tim went to the park to play. He saw a big red ball that story caused the dishwasher.

**chat-v6-scratch** [hit]

> Once upon a time, there was a little bunny named Bobo. Bobo lived in a small house with his best friend, a little girl named Sue. Sue loved to play and run all day. / One day, Bunny was playing near a big rock. He saw a big buckle in the ground. Sue wanted to get the bunny. Sue said, "Hi, Turtle!
