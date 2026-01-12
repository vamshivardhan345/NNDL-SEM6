import math



# Inputs (free, win, offer)

inputs = [1, 0, 1]



# Weights (Input → Hidden)

weights_hidden = [

  [0.5, -0.2, 0.3],  # H1

  [0.4, 0.1, -0.5]  # H2

]



# Bias for hidden layer

bias_hidden = [0.1, -0.1]



# Weights (Hidden → Output)

weights_output = [0.7, 0.2]



# Bias for output layer

bias_output = 0.05



# Sigmoid function

def sigmoid(x):

  return 1 / (1 + math.exp(-x))



# -------- Forward Propagation --------



# Hidden layer

hidden_inputs = []

hidden_outputs = []



for i in range(len(weights_hidden)):

  total = 0

  for j in range(len(inputs)):

    total += inputs[j] * weights_hidden[i][j]

  total += bias_hidden[i]

  hidden_inputs.append(total)

  hidden_outputs.append(sigmoid(total))



# Output layer

output_sum = 0

for i in range(len(hidden_outputs)):

  output_sum += hidden_outputs[i] * weights_output[i]



output_sum += bias_output

spam_probability = sigmoid(output_sum)



# Output

print("Hidden Layer Inputs :", hidden_inputs)

print("Hidden Layer Outputs:", hidden_outputs)

print("Spam Probability  :", spam_probability)



OUTPUT:

Hidden Layer Inputs : [0.9, -0.19999999999999998]
Hidden Layer Outputs: [0.7109495026250039, 0.45016600268752216]
Spam Probability   : 0.6542328717521364
