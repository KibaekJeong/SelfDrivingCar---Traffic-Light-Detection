import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
import os

flags = tf.app.flags
flags.DEFINE_string('model_path',None,'Path of the frozen model to be optimized')
flags.DEFINE_string('output_path',None,'Path of output model to be saved')
tf.app.flags.mark_flag_as_required('model_path')

FLAGS = flags.FLAGS

def load_graph(path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(path,'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def,name='')
    return graph_def

def stats(graph_def):
    print('\nInput Feature Nodes: {}'.format([node.name for node in graph_def.node if node.op=='Placeholder']))
    print('Output Nodes: {}'.format([node.name for node in graph_def.node if ('detection' in node.name)]))
    print('Constant Count: {}'.format(len([node for node in graph_def.node if node.op=='Const'])))
    print('Identity Count: {}'.format(len([node for node in graph_def.node if node.op=='Identity'])))
    print('Total nodes: {}'.format(len(graph_def.node)))

def optimize_graph(model_file,output_dir):
    input_names = ['image_tensor']
    output_names = ['num_detections', 'detection_classes', 'detection_scores', 'detection_boxes']
    graph_def = load_graph(model_file)

    print('Graph Statistics Before Optimization')
    stats(graph_def)
    print('------------------------------------')
    transforms = [
        'strip_unused_nodes(type=float, shape="1,299,299,3")',
        'remove_nodes(op=Identity)',
        'fold_constants(ignore_errors=true)',
        'fold_batch_norms',
        'fold_old_batch_norms',
    ]
    optimized_graph_def = TransformGraph(graph_def,input_names,output_names,transforms)
    print('\nGraph Statistics After Optimization')
    stats(optimized_graph_def)
    print('------------------------------------')

    tf.train.write_graph(optimized_graph_def,logdir=output_dir,as_text=False,name='optimized_frozen_model.pb')

def main(unused_argv):
    if FLAGS.output_path is None:
        FLAGS.output_path = os.path.join(os.path.dirname(FLAGS.model_path), 'optimized_model')

    optimize_graph(FLAGS.model_path,FLAGS.output_path)

if __name__=='__main__':
    tf.app.run()
