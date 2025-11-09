

 are implementing the function that provides the targets (which is different between different methods: flow matching, shortcut models, consistency models,...):

\[
(x_t, v_t, t, k) = \text{get\_targets(method=flow matching/shortcut/\dots)}.
\]

$k$ is the level code parameter used to encode $dt$ (the step size), it is used in certain methods only.

The rest of the code is the same where DiT model inputs $x_t, t$ (and optionally k) and outputs ${\hat{v_t}}$.
The loss would be $mse(v_t, {\hat{v_t}})$.
