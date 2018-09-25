import numpy as np
import tensorflow as tf
from PIL import Image
import ctc_joint_seq2seq_attention_mjsynth

flags = tf.flags
flags.DEFINE_string('frozen_graph_path', r'Y:\ocr4\data\trade_model_pb\frozen_inference_graph.pb', 'Path to model frozen graph.')
FLAGS = flags.FLAGS

def caption_image():
    image = Image.open(r'Y:\ocr4\res_2\cropped_2.jpg')
    image = image.resize((310, 32), Image.ANTIALIAS)
    image = np.array(image)
    width = image.shape[1]
    image = np.reshape(image, (32, -1, 1))
    image = np.reshape(image, [1, 32, -1, 1])
    image = image / 255.0 - 0.5
    return image, width

def _get_string(labels):
    """Transform an 1D array of labels into the corresponding character string"""
    string = ''.join([ctc_joint_seq2seq_attention_mjsynth.out_charset[c] for c in labels])
    return string

def main(_):
    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with model_graph.as_default():
        with tf.Session(graph=model_graph) as sess:
            inputs = model_graph.get_tensor_by_name('image_tensor:0')
            inputs_width = model_graph.get_tensor_by_name('image_width_tensor:0')
            classes = model_graph.get_tensor_by_name('predict:0')
            image, image_width = caption_image()
            predicted_label = sess.run(classes,
                                       feed_dict={inputs: image, inputs_width: image_width})

            print(_get_string(predicted_label))


if __name__ == '__main__':
    tf.app.run()
