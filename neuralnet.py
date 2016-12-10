import tensorflow as tf

# Parameters
learning_rate = 0.05
training_epochs = 75
display_step = 1
batch_size = 5000

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 833 # Chess position array: 64 squares + board.turn
n_classes = 1

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
cost = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# --------------------- File Parsing -------------------------------

file = open("train.txt", "r")
lines = file.readlines()
X = []; Y = []
total_lines = 1000000#len(lines)

for l in range(total_lines):
    print "Progress: %i / %i" % (l + 1,  total_lines) 
    line = lines[l]
    line = line.split("[")
    Y.append(int(line[0]))
    line = (line[1])[:-2]
    line = line.split(",")
    position = []

    for i in range(len(line)):
        n = float(line[i])
        position.append(n)
    X.append(position)

file.close()
assert(len(X) == len(Y))
Y = map(list, zip(*[Y]))

batches_x = []
batches_y = []

for i in range(len(X)/batch_size):
    if i == 0:
        batches_x.append(X[:batch_size])
        batches_y.append(Y[:batch_size])
    else:
        batches_x.append(X[(batch_size * (i - 1)):(batch_size * i)])
        batches_y.append(Y[(batch_size * (i - 1)):(batch_size * i)])

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        total_batch = len(batches_x) #3875492
        i = 0
        for i in range(total_batch):
            batch_x = batches_x[i]
            batch_y = batches_y[i]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            i += batch_size
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    save_path = saver.save(sess, './chess-ann.ckpt')
    print("Model saved in file: %s" % save_path)

    #------------------- Testing --------------------------------

    test_file = open("test.txt", "r")
    tests = test_file.readlines()
    test_y = []
    test_x = []

    for test in tests:
        test = test.split("[")
        test_y.append(int(test[0]))
        test = (test[1])[:-2]
        test = test.split(",")
        position = []
        for i in range(len(test)):
            if test[i] == " ":
                print test_y
            n = float(test[i])
            position.append(n)
        test_x.append(position)

    test_y = [test_y]
    test_y = map(list, zip(*test_y))
    
    assert(len(test_x) == len(test_y))

    #print "Accuracy:", accuracy.eval({x: test_x, y: test_y})

    _, test_c = sess.run([optimizer, cost], feed_dict={x: test_x, y: test_y})
    test_cost = test_c / len(test_x)
    print "Testing cost = {:.9f}".format(test_cost)
