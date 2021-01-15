import numpy as np
import tensorflow as tf
import os, shutil

# Input (2D)
x = np.array([[x1, x2] for x1 in np.linspace(10, 20, 4) for x2 in np.linspace(-7, -3, 3)])
# Output (3D)
f = np.array([[np.sin(np.sum(xx)), np.cos(np.sum(xx)), np.cos(np.sum(xx)) ** 2] for xx in x])

# Dimension of input x and output f
d_x = x.shape[-1]
d_f = f.shape[-1]

# Placeholders
x_p = tf.placeholder(tf.float64, [None, d_x], 'my_x_p')
f_p = tf.placeholder(tf.float64, [None, d_f], 'my_f_p')

# Model
model = x_p
model = tf.layers.dense(model, 7, tf.tanh)
model = tf.layers.dense(model, 5, tf.tanh)
model = tf.layers.dense(model, d_f, None)
model = tf.identity(model, 'my_model')

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Evaluate for later check of serving
f_model = sess.run(model, {x_p: x})
folder = 'data'
if not os.path.exists(folder):
    os.mkdir(folder)
np.savetxt('data/x.dat', f_model)
np.savetxt('data/f_model.dat', f_model)

# Save model
model_path = 'saved/model/001'
saver = tf.saved_model.builder.SavedModelBuilder(model_path)


info_input = tf.saved_model.utils.build_tensor_info(x_p)
info_output = tf.saved_model.utils.build_tensor_info(model)
signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs={'x': info_input}
    , outputs={'f': info_output},
    # method_name = "tensorflow/serving/predict"
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
)
saver.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                   signature_def_map={'serving_default': signature})
saver.save()

sess.close()
tf.reset_default_graph()