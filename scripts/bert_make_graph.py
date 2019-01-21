assert __name__ == "__main__", "This script is not designed to be imported"
from bert_path import BERT_CODE_PATH
# bert version used: https://github.com/google-research/bert/tree/f39e881b169b9d53bea03d2d341b31707a6c052b
# BERT_CODE_PATH should be the parent folder of the repo, so "import bert" works

import sys, os
sys.path.append(os.path.expanduser(BERT_CODE_PATH))

import tensorflow as tf
import bert
import bert.modeling, bert.optimization

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bert_model_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uncased_L-12_H-768_A-12"))
parser.add_argument("--bert_output_file")
parser.add_argument("--feature_downscale", type=float, default=4.0)
args = parser.parse_args()

if not args.bert_output_file:
    model_name = os.path.split(os.path.split(args.bert_model_dir)[0])[1]
    # e.g. uncased_L-12_H-768_A-12_FDS-4.0_graph.pb
    args.bert_output_file = os.path.join("bert_models", os.path.basename("{}_FDS-{}_graph.pb".format(model_name, args.feature_downscale)))


sess = tf.InteractiveSession()

# %%

config = bert.modeling.BertConfig.from_json_file(os.path.join(args.bert_model_dir, "bert_config.json"))

# %%

input_ids = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_ids')
word_end_mask = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_end_mask')
is_training = tf.placeholder(shape=(), dtype=tf.bool, name='is_training')

# %%

input_mask = (1 - tf.cumprod(1 - word_end_mask, axis=-1, reverse=True))
token_type_ids = tf.zeros_like(input_ids)

# %%

def dropout_only_if_training(input_tensor, dropout_prob):
    """
    A replacement for bert.modeling.dropout that uses is_training to determine
    whether to perform dropout.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    keep_prob = tf.cond(is_training, lambda: 1.0 - dropout_prob, lambda: 1.0)
    output = tf.nn.dropout(input_tensor, keep_prob)
    return output

bert.modeling.dropout = dropout_only_if_training

# %%

model = bert.modeling.BertModel(config=config,
                                is_training=True,  # We disable dropout ourselves, based on a placeholder
                                input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

def get_word_features():
    subword_features = model.get_sequence_output()
    word_features_packed = tf.gather(
        tf.reshape(subword_features, [-1, int(subword_features.shape[-1])]),
        tf.to_int32(tf.where(tf.reshape(word_end_mask, (-1,))))[:,0])

    # XXX(nikita): this division by 4 should not be hard-coded, and it should
    # be implemented in the c++ portion of the network (not here)
    # The idea behind rescaling is that the code mixes BERT vectors and vectors
    # that are output by an LSTM, which would be a magnitude mismatch without
    # any rescaling.
    print("RESCALING word features: dividing by {}".format(args.feature_downscale))
    word_features_packed = word_features_packed / args.feature_downscale

    # input_mask is over subwords, whereas valid_mask is over words
    sentence_lengths = tf.reduce_sum(word_end_mask, -1)
    valid_mask = (tf.range(tf.reduce_max(sentence_lengths))[None,:] < sentence_lengths[:, None])
    dim_padded = tf.shape(valid_mask)[:2]
    mask_flat = tf.reshape(valid_mask, (-1,))
    dim_flat = tf.shape(mask_flat)[:1]
    nonpad_ids = tf.to_int32(tf.where(mask_flat)[:,0])
    word_features_padded = tf.scatter_nd(
        indices=nonpad_ids[:, None],
        updates=word_features_packed,
        shape=tf.concat([dim_flat, tf.shape(word_features_packed)[1:]], axis=0)
        )
    word_features_padded = tf.reshape(
        word_features_padded,
        tf.concat([dim_padded, [-1]], axis=0)
        )
    return word_features_padded

word_features = tf.identity(get_word_features(), name='word_features')
word_features_grad = tf.placeholder(shape=(None, None, word_features.shape[-1]), dtype=word_features.dtype, name='word_features_grad')

# %%

saver = tf.train.Saver()

# %% verify that restoring a checkpoint works

saver.restore(sess, os.path.join(args.bert_model_dir, "bert_model.ckpt"))

# %%

def conditional_print(do_print, tensor, message):
    return tf.cond(do_print,
        lambda: tf.Print(tensor, [tensor], message),
        lambda: tensor
        )

def conditional_print_norm(do_print, tensor, message):
    return tf.cond(do_print,
        lambda: tf.Print(tensor, [tf.norm(tensor)], message),
        lambda: tensor
        )

def create_optimizer(ys, grad_ys, init_lr=5e-5, num_warmup_steps=160):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    with sess.graph.as_default() as g, g.name_scope(None):
        learning_rate_var = tf.get_variable(
            "learning_rate",
            shape=(),
            dtype=tf.float32,
            initializer=tf.constant_initializer(init_lr),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        warmup_steps_var = tf.get_variable(
            "warmup_steps",
            shape=(),
            dtype=tf.int32,
            initializer=tf.constant_initializer(num_warmup_steps, dtype=tf.int32),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])

    learning_rate = learning_rate_var

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = warmup_steps_var

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = learning_rate_var * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
    (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = bert.optimization.AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        # weight_decay_rate=0.01,
        # MODIFIED(nikita): the original weight decay value caused parameters to
        # all decay to zero in our training setup.
        weight_decay_rate=0.00,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    print_every = 12
    do_print = (global_step < print_every) | tf.equal(
        global_step % print_every, print_every - 1)

    tvars = tf.trainable_variables()

    grad_ys = [
        conditional_print_norm(
            do_print,
            tf.check_numerics(node, "check_numerics failed for placeholder {}".format(node.name.split(':')[0])),
            "Norm for value passed to placeholder {}: ".format(node.name.split(':')[0]),
            )
        for node in grad_ys
        ]
    # grads = tf.gradients(loss, tvars)
    grads = tf.gradients(ys=ys, xs=tvars, grad_ys=grad_ys)

    # This is how the model was pre-trained.
    global_norm = tf.global_norm(grads)
    global_norm = conditional_print(
        do_print,
        global_norm,
        "Gradient norm for BERT parameters: "
        )
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0, use_norm=global_norm)

    train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)], name='train')
    return train_op, learning_rate_var, warmup_steps_var

# %%

train_op, learning_rate_var, warmup_steps_var = create_optimizer([word_features], [word_features_grad])

# %%

init_op = tf.variables_initializer(tf.global_variables(), name='init')

# %%

new_learning_rate = tf.placeholder(shape=(), dtype=tf.float32, name='new_learning_rate')
new_warmup_steps = tf.placeholder(shape=(), dtype=tf.int32, name='new_warmup_steps')

set_learning_rate_op = tf.assign(learning_rate_var, new_learning_rate, name='set_learning_rate')
set_warmup_steps_op = tf.assign(warmup_steps_var, new_warmup_steps, name='set_warmup_steps')

# %%

print(f"""
Operation names:

input_ids: {input_ids.name}
word_end_mask: {word_end_mask.name}
is_training: {is_training.name}
word_features: {word_features.name}
word_features_grad: {word_features_grad.name}
init_op: {init_op.name}
new_learning_rate: {new_learning_rate.name}
set_learning_rate_op: {set_learning_rate_op.name}
new_warmup_steps: {new_warmup_steps.name}
set_warmup_steps_op: {set_warmup_steps_op.name}
train_op: {train_op.name}
save_op: {saver.saver_def.save_tensor_name}
restore_op: {saver.saver_def.restore_op_name}
checkpoint_name: {saver.saver_def.filename_tensor_name}
""")

# %%

with open(args.bert_output_file, 'wb') as f:
    f.write(sess.graph_def.SerializeToString())

print("Saved tensorflow graph to to", args.bert_output_file)

# %%
