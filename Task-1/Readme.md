# Task-1

## Transformers
Implementation of the transformer architecture described in the paper can be found in `Transformer.py`.


The "Attention Is All You Need" paper introduced the Transformer,
a novel architecture for sequence-to-sequence tasks that ditches the recurrent and convolutional layers prevalent in LSTMs and GRUs. 
Instead, it relies solely on an attention mechanism. This mechanism analyzes the relationships between elements in a sequence, allowing the Transformer to capture long-range dependencies more effectively. 
Unlike LSTMs and GRUs, which process information sequentially, the Transformer can analyze all parts of the sequence simultaneously. 
This parallelism makes it faster to train and potentially more powerful, leading to superior performance in machine translation and other tasks.

## Results

After training the model for 5 epochs, it achieved accuracy of 98.86% on train dataset and 99.01% on validation dataset.
