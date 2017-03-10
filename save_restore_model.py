import tensorflow as tf

def save_model(w1_shape, w2_shape, model_name, collection_name):
	w1 = tf.Variable(tf.truncated_normal(w1_shape), name='w1')
	w2 = tf.Variable(tf.truncated_normal(w2_shape), name='w2')
	tf.add_to_collection(collection_name, w1)
	tf.add_to_collection(collection_name, w2)
	saver = tf.train.Saver()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	saver.save(sess, model_name)

def restore_mode(model_name, collection_name):
	sess = tf.Session()
	graph = model_name + '.meta'
	new_saver = tf.train.import_meta_graph(graph)
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
	all_vars = tf.get_collection(collection_name)
	for v in all_vars:
    	v_ = sess.run(v)
    	print(v_)

