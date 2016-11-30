import tensorflow as tf

sess = tf.Session()
new_saver = tf.train.import_meta_graph('chess-ann.meta')
new_saver.restore(sess, 'chess-ann.data-00000-of-00001')
all_vars = tf.trainable_variables()


