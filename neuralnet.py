import numpy as np
import tensorflow as tf

file = open("train.txt", "r")
lines = file.readlines()

X = []
Y = []

def classify(num):
    result = [0 for _ in range(200)]
    if num in range(-1200, 1200):
        if num < 0:
            num = int((float(num) / 1200) * 100)
            result[99 - num] = 1
        else:
            num = (float(num) / 1200) * 100
            result[int(num)] = 1
    return result


for line in lines:
    line = line.split("[")
    Y.append(classify(int(line[0])))
    line = (line[1])[:-2]
    line = line.split(",")
    position = []
    for i in range(len(line)):
        n = float(line[i])
        position.append(n)
    X.append(position)

assert(len(X) == len(Y))
#print "size", len(X)

batches_x = []
batches_y = []

for i in range(len(X)/100):
    if i == 0:
        batches_x.append(X[:100])
        batches_y.append(Y[:100])
    else:
        batches_x.append(X[(100*(i-1)):(100*i)])
        batches_y.append(Y[(100*(i-1)):(100*i)])

"""
Each line in lines[] looks like:
18
[0.2, 0.3, 0.4, 0.5, 0.6, 0.4, 0, 0.2, 
0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
0, 0, 0, 0, 0, 0.3, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 
-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 
-0.2, -0.3, -0.4, -0.5, -0.6, -0.4, -0.3, -0.2, 
0]

Here, X = [0.2, ..., 0] and y = 18
"""

# Parameters
learning_rate = 0.05
training_epochs = 3
display_step = 1
batch_size = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 65 # Chess position array: 64 squares + board.turn
n_classes = 200 # Possible engine scores (-10,000 < score < 10,000)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        total_batch = len(batches_x)
        for i in range(total_batch):
            batch_x = batches_x[i]
            batch_y = batches_y[i]

            #batch_x = np.reshape(batch_x, (65, 1))
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver.save(sess, 'chess-ann')

    test_file = open("test.txt", "r")
    tests = test_file.readlines()
    test_y = []
    test_x = []

    for test in tests:
        test = test.split("[")
        test_y.append(classify(int(test[0])))
        test = (test[1])[:-2]
        test = test.split(",")
        position = []
        for i in range(len(test)):
            n = float(test[i])
            position.append(n)
        test_x.append(position)
        
    print "Accuracy:", accuracy.eval({x: test_x, y: test_y})
    test_var = [0, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0.5, 0, 0.1, 0, 0, 0, 0, 0, 0.1, 0, 0.4, 0.1, 0, 0, 0, 0.1, -0.1, 0, 0, -0.1, 0, 0, 0.1, -0.1, 0, 0, -0.1, 0, 0, 0.1, -0.1, 0, 0, 0, 0, 0, 0, -0.1, 0, 0, 0, -0.5, -0.6, 0, 0, 0, 0, 0, 0, -0.2, 0, 0] 
    
    print sess.run(tf.argmax(pred,1), feed_dict = {x: [test_var]})
    print classify(-287)