<p>
OK so h now we have to think about how we are going to implement this NN

So in all my previous approaches in language models each one of my tokens was a word input. And this word was known in the vocabullary. 

But in this case the context words are words in our vocabullary, but it is the case that our missspelt word is actually not in our vocabullary.

So how will we choose to represent these words as vectors

1. We could try to predict most commonly misspelt words, but how is that any better than hardcoded examples, actually it would be worse

2. We could try to encode the individual characters as vectors. But this would make it harder for the NN to create actual relationships between the words.

Is it the case that we can have an encoder which has learned to take the words as vec inputs and then we can take those context units into a decoder which has learned the characters as embeddings. 

And the deocder also has to pick a word from our vocab to output through the whole softmax thing. lol

___

So we can actually do a ranking system where we take the cossine simularity between 



<p>