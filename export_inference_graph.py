import tensorflow as tf

import exporter
from model_class import CRNN_MODEL

slim = tf.contrib.slim
flags = tf.flags

flags.DEFINE_string('input_type1', 'image_tensor', 'Type of input node. Can '
                                                  "be one of ['image_tensor', 'image_width_tensor']")
flags.DEFINE_string('input_type2', 'image_width_tensor', 'Type of input node. Can '
                                                  "be one of ['image_tensor', 'image_width_tensor']")
flags.DEFINE_string('input_shape', None, "If input_type is 'image_tensor', "
                                         "this can be explicitly set the shape of this input "
                                         "to a fixed size. The dimensions are to be provided as a "
                                         "comma-seperated list of integers. A value of -1 can be "
                                         "used for unknown dimensions. If not specified, for an "
                                         "'image_tensor', the default shape will be partially "
                                         "specified as '[None, None, None, 3]'.")
flags.DEFINE_string('trained_checkpoint_prefix', '../data/trade_model31/model.ckpt-500526',
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('output_directory', '../data/trade_model_pb', 'Path to write outputs')
# tf.flags.mark_flag_as_required('trained_checkpoint_prefix')
# tf.flags.mark_flag_as_required('output_directory')
FLAGS = flags.FLAGS


def main(_):
    model = CRNN_MODEL()
    input_shape = None
    input_width_shape = None
    exporter.export_inference_graph(FLAGS.input_type1,
                                    FLAGS.input_type2,
                                    model,
                                    FLAGS.trained_checkpoint_prefix,
                                    FLAGS.output_directory,
                                    input_shape,
                                    input_width_shape)

if __name__ == '__main__':
    tf.app.run()
