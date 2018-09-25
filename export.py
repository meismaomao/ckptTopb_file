"""
Created on 2018-9-23
@author: qlyuan
"""

"""
Function to compress .ckpt file and export CKPT to PB file.

Modified from: Tensorflow models/research/object_detection/export.py

"""

import logging
import os
import tempfile
import tensorflow as tf

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib

slim = tf.contrib.slim

def freeze_graph_with_def_protos(input_graph_def,
                                 input_saver_def,
                                 input_checkpoint,
                                 output_node_names,
                                 restore_op_name,
                                 filename_tensor_name,
                                 clear_devices,
                                 initializer_nodes,
                                 variable_names_blacklist=''):

    del restore_op_name, filename_tensor_name

    if not saver_lib.checkpoint_exists(input_checkpoint):
        raise ValueError(
            "Input checkpoint" + input_checkpoint + "does not exist!"
        )

    if not output_node_names:
        raise ValueError(
            "You must supply the name of a node to --output_node_names."
        )

    if clear_devices:
        for node in input_graph_def.node:
            node.device = ''

    with tf.Graph().as_default():
        tf.import_graph_def(input_graph_def, name='')
        config = tf.ConfigProto(graph_options=tf.GraphOptions())
        with session.Session(config=config) as sess:
            if input_saver_def:
                saver = saver_lib.Saver(saver_def=input_saver_def)
                saver.restore(sess, input_checkpoint)
            else:
                var_list = {}
                reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
                var_to_shape_map = reader.get_variable_to_dtype_map()
                for key in var_to_shape_map:
                    try:
                        tensor = sess.graph.get_tensor_by_name(key + ':0')
                    except KeyError:
                        continue
                    var_list[key] = tensor
                saver = saver_lib.Saver(var_list=var_list)
                saver.restore(sess, input_checkpoint)
                if initializer_nodes:
                    sess.run(initializer_nodes)
            variable_names_blacklist = (variable_names_blacklist.split(',') if
                                        variable_names_blacklist else None)
            output_graph_def = graph_util.convert_variables_to_constants(sess,
                                                                         input_graph_def,
                                                                         output_node_names.split(','),
                                                                         variable_names_blacklist
                                                                         =variable_names_blacklist)
        return output_graph_def

def replace_variable_values_with_moving_average(graph,
                                                current_checkpoint_file,
                                                new_checkpoint_file,
                                                decay=0.0):
    with graph.as_default():
        variable_average = tf.train.ExponentialMovingAverage(decay)
        ema_variables_to_restore = variable_average.variables_to_restore()
        with tf.Session() as sess:
            read_saver = tf.train.Saver(ema_variables_to_restore)
            read_saver.restore(sess, current_checkpoint_file)
            writer_saver = tf.train.Saver()
            writer_saver.save(sess, new_checkpoint_file)

def _image_tensor_input_placeholder(input_shape=None):
    if input_shape is None:
        input_shape = (None, 32, None, 1)
    input_tensor = tf.placeholder(dtype=tf.float32, shape=input_shape, name='image_tensor')
    return input_tensor, input_tensor

def _image_width_tensor_input_placeholder(input_shape=None):
    if input_shape is None:
        input_shape = []
    input_tensor = tf.placeholder(dtype=tf.int32, shape=input_shape, name='image_width_tensor')
    return input_tensor, input_tensor

input_placeholder_fn_map = {'image_tensor': _image_tensor_input_placeholder,
                            'image_width_tensor': _image_width_tensor_input_placeholder}

def _add_tensor_output_nodes(postprocessed_tensors, output_collection_name='inference_op'):
    outputs = {}
    predict = postprocessed_tensors.get('predictions')
    outputs['predict'] = tf.identity(predict, name='predict')
    for output_key in outputs:
        tf.add_to_collection(output_collection_name, outputs[output_key])
    return outputs

def write_frozen_graph(frozen_graph_path, frozen_graph_def):
    with gfile.GFile(frozen_graph_path, 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
    logging.info('%d ops in the final graph.', len(frozen_graph_def.node))

def write_saved_model(saved_model_path,
                      frozen_graph_def,
                      inputs,
                      inputs_width,
                      outputs):
    with tf.Graph().as_default():
        with session.Session() as sess:
            tf.import_graph_def(frozen_graph_def, name='')
            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
            tensor_info_inputs = {'inputs': tf.saved_model.utils.build_tensor_info(inputs),
                                  'inputs_width': tf.saved_model.utils.build_tensor_info(inputs_width)}
            tensor_info_outputs = {}

            for k, v in outputs.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)
            detection_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                inputs=tensor_info_inputs,
                outputs=tensor_info_outputs,
                method_name=signature_constants.PREDICT_METHOD_NAME
            ))

            builder.add_meta_graph_and_variables(sess,
                                                 [tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                                    detection_signature, },)
            builder.save()


def write_graph_and_checkpoint(inference_graph_def,
                               model_path,
                               input_saver_def,
                               trained_checkpoint_prefix):
    for node in inference_graph_def.node:
        node.device = ''

    with tf.Graph().as_default():
        tf.import_graph_def(inference_graph_def, name='')
        with session.Session() as sess:
            saver = saver_lib.Saver(saver_def=input_saver_def,
                                    save_relative_paths=True)
            saver.restore(sess, trained_checkpoint_prefix)
            saver.save(sess, model_path)

def _get_outputs_from_inputs(input_tensors, input_width_tensors, model, output_collection_name):

    inputs = tf.to_float(input_tensors)
    inputs_width_tensors = tf.to_int32(input_width_tensors)
    preprocessed_inputs, preprocessed_inputs_width = model.convnet_layers(inputs, inputs_width_tensors)
    output_tensors = model.rnn_layers(preprocessed_inputs, preprocessed_inputs_width)
    prediction = model.post_processed(output_tensors, preprocessed_inputs_width)

    return _add_tensor_output_nodes(prediction, output_collection_name)

def _build_model_graph(input_type1, input_type2, model, input_shape1, input_shape2, output_collection_name, graph_hook_fn):
    if input_type1 not in input_placeholder_fn_map:
        raise ValueError('Unknown in type: {}'.format(input_type1))

    if input_type2 not in input_placeholder_fn_map:
        raise ValueError('Unknown in type: {}'.format(input_type2))

    placeholder_args1 = {}
    placeholder_args2 = {}

    if input_shape1 is not None:
        if input_type1 != 'image_tensor':
            raise ValueError("Can only specify input shape for 'image_tensor input'")
        placeholder_args1['input_shape'] = input_shape1

    if input_shape2 is not None:
        if input_type2 != 'image_width_tensor':
            raise ValueError("Can only specify input shape for 'image_tensor input'")
        placeholder_args2['input_width_shape'] = input_shape2

    placeholder_tensor1, input_tensors1 = input_placeholder_fn_map[input_type1](
        **placeholder_args1)

    placeholder_tensor2, input_tensors2 = input_placeholder_fn_map[input_type2](
        **placeholder_args2)

    outputs = _get_outputs_from_inputs(
        input_tensors=input_tensors1,
        input_width_tensors=input_tensors2,
        model=model,
        output_collection_name=output_collection_name)

    # Add global step to the graph
    slim.get_or_create_global_step()

    if graph_hook_fn:
        graph_hook_fn()

    return outputs, placeholder_tensor1, placeholder_tensor2


def export_inference_graph(input_type1,
                           input_type2,
                           model,
                           trained_checkpoint_prefix,
                           output_dir,
                           input_shape1=None,
                           input_shape2=None,
                           use_moving_averages=None,
                           output_collection_name='inference_op',
                           additional_output_tensor_names=None,
                           graph_hook_fn=None):
    tf.gfile.MakeDirs(output_dir)
    frozen_graph_path = os.path.join(output_dir, 'frozen_inference_graph.pb')
    saved_model_path = os.path.join(output_dir, 'saved_model')
    model_path = os.path.join(output_dir, 'model.ckpt')

    outputs, placeholder_tensor1, placeholder_tensor2 = _build_model_graph(input_type1=input_type1,
                                                                           input_type2=input_type2,
                                                                           model=model,
                                                                           input_shape1=input_shape1,
                                                                           input_shape2=input_shape2,
                                                                           output_collection_name=output_collection_name,
                                                                           graph_hook_fn=graph_hook_fn)

    saver_kwargs = {}

    if use_moving_averages:
        # This check is to be compatible with both version of SaverDef.
        if os.path.isfile(trained_checkpoint_prefix):
            saver_kwargs['write_version'] = saver_pb2.SaverDef.V1
            temp_checkpoint_prefix = tempfile.NamedTemporaryFile().name
        else:
            temp_checkpoint_prefix = tempfile.mkdtemp()
            replace_variable_values_with_moving_average(
            tf.get_default_graph(), trained_checkpoint_prefix,
            temp_checkpoint_prefix)
        checkpoint_to_use = temp_checkpoint_prefix
    else:
        checkpoint_to_use = trained_checkpoint_prefix

    saver = tf.train.Saver(**saver_kwargs)
    input_saver_def = saver.as_saver_def()

    write_graph_and_checkpoint(
        inference_graph_def=tf.get_default_graph().as_graph_def(),
        model_path=model_path,
        input_saver_def=input_saver_def,
        trained_checkpoint_prefix=checkpoint_to_use)

    if additional_output_tensor_names is not None:
        output_node_names = ','.join(outputs.keys() + additional_output_tensor_names)
    else:
        output_node_names = ','.join(outputs.keys())

    frozen_graph_def = freeze_graph_with_def_protos(
        input_graph_def=tf.get_default_graph().as_graph_def(),
        input_saver_def=input_saver_def,
        input_checkpoint=checkpoint_to_use,
        output_node_names=output_node_names,
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        clear_devices=True,
        initializer_nodes='')
    write_frozen_graph(frozen_graph_path, frozen_graph_def)
    write_saved_model(saved_model_path, frozen_graph_def,
                      placeholder_tensor1, placeholder_tensor2, outputs)
