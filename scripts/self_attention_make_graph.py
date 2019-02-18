assert __name__ == "__main__", "This script is not designed to be imported"
from bert_path import BERT_CODE_PATH
# bert version used: https://github.com/google-research/bert/tree/f39e881b169b9d53bea03d2d341b31707a6c052b
# BERT_CODE_PATH should be the parent folder of the repo, so "import bert" works

import sys, os
sys.path.append(os.path.expanduser(BERT_CODE_PATH))

import numpy as np
import tensorflow as tf
import bert
import bert.modeling, bert.optimization

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bert_model_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uncased_L-12_H-768_A-12"))
parser.add_argument("--bert_output_file")
parser.add_argument("--feature_downscale", type=float, default=1.0)
parser.add_argument("--num_nonbert_layers", type=int, default=0)
parser.add_argument("--disable_bert", action="store_true")
parser.add_argument("--nonbert_vocabulary_size", type=int, default=0)
args = parser.parse_args()

if not args.bert_output_file:
    model_name = os.path.split(os.path.split(args.bert_model_dir)[0])[1]
    # e.g. uncased_L-12_H-768_A-12_FDS-4.0_graph.pb
    if args.feature_downscale != 1.0:
        model_name += "_FDS-{}".format(args.feature_downscale)
    model_name = "{}_graph.pb".format(model_name)
    args.bert_output_file = os.path.join("bert_models", os.path.basename(model_name))

# %%

sess = tf.InteractiveSession()

# %%

if args.disable_bert:
    config = None
else:
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

if args.disable_bert:
    model = None
    bert_tvars = []
else:
    model = bert.modeling.BertModel(config=config,
                                    is_training=True,  # We disable dropout ourselves, based on a placeholder
                                    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
    bert_tvars = tf.trainable_variables()

# %%

if args.num_nonbert_layers > 0:
    position_table = tf.get_variable('position_table', shape=[500, 512], initializer=tf.initializers.random_normal())
else:
    position_table = None

# %%

HPARAMS = {
    'attention_dropout': 0.2,
    'relu_dropout': 0.1,
    'residual_dropout': 0.2,
}

def make_layer_norm(input, torch_name, name):
    return tf.contrib.layers.layer_norm(
        inputs=input, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def make_heads(input, shape_bthf, shape_xtf, torch_name, name):
    res = tf.layers.dense(input, 8 * 64 // 2, activation=None, name=name,
                use_bias=False,
                kernel_initializer=tf.glorot_normal_initializer())
    res = tf.reshape(res, shape_bthf)
    res = tf.transpose(res, (0,2,1,3)) # batch x num_heads x time x feat
    res = tf.reshape(res, shape_xtf) # _ x time x feat
    return res

def make_attention(input, nonpad_ids, dim_flat, dim_padded, valid_mask, torch_name, name):
    input_flat = tf.scatter_nd(indices=nonpad_ids[:, None], updates=input, shape=tf.concat([dim_flat, tf.shape(input)[1:]], axis=0))
    input_flat_dat, input_flat_pos = tf.split(input_flat, 2, axis=-1)

    shape_bthf = tf.concat([dim_padded, [8, -1]], axis=0)
    shape_bhtf = tf.convert_to_tensor([dim_padded[0], 8, dim_padded[1], -1])
    shape_xtf = tf.convert_to_tensor([dim_padded[0] * 8, dim_padded[1], -1])
    shape_xf = tf.concat([dim_flat, [-1]], axis=0)

    qs1 = make_heads(input_flat_dat, shape_bthf, shape_xtf, f'{torch_name}.w_qs1', f'{name}/q_dat')
    ks1 = make_heads(input_flat_dat, shape_bthf, shape_xtf, f'{torch_name}.w_ks1', f'{name}/k_dat')
    vs1 = make_heads(input_flat_dat, shape_bthf, shape_xtf, f'{torch_name}.w_vs1', f'{name}/v_dat')
    qs2 = make_heads(input_flat_pos, shape_bthf, shape_xtf, f'{torch_name}.w_qs2', f'{name}/q_pos')
    ks2 = make_heads(input_flat_pos, shape_bthf, shape_xtf, f'{torch_name}.w_ks2', f'{name}/k_pos')
    vs2 = make_heads(input_flat_pos, shape_bthf, shape_xtf, f'{torch_name}.w_vs2', f'{name}/v_pos')

    qs = tf.concat([qs1, qs2], axis=-1)
    ks = tf.concat([ks1, ks2], axis=-1)
    attn_logits = tf.matmul(qs, ks, transpose_b=True) / (1024 ** 0.5)

    attn_mask = tf.reshape(tf.tile(valid_mask, [1,8*dim_padded[1]]), tf.shape(attn_logits))
    # TODO(nikita): use tf.where and -float('inf') here?
    attn_logits -= 1e10 * tf.to_float(~attn_mask)

    attn = tf.nn.softmax(attn_logits)
    attn = dropout_only_if_training(attn, HPARAMS['attention_dropout'])

    attended_dat_raw = tf.matmul(attn, vs1)
    attended_dat_flat = tf.reshape(tf.transpose(tf.reshape(attended_dat_raw, shape_bhtf), (0,2,1,3)), shape_xf)
    attended_dat = tf.gather(attended_dat_flat, nonpad_ids)
    attended_pos_raw = tf.matmul(attn, vs2)
    attended_pos_flat = tf.reshape(tf.transpose(tf.reshape(attended_pos_raw, shape_bhtf), (0,2,1,3)), shape_xf)
    attended_pos = tf.gather(attended_pos_flat, nonpad_ids)

    stdv = 1 / np.sqrt(8 * 64 // 2)
    attended_dat.set_shape((None, 8 * 64 // 2))
    attended_pos.set_shape((None, 8 * 64 // 2))

    out_dat = tf.layers.dense(attended_dat, 512, activation=None, name=f'{name}/proj1',
                use_bias=False,
                kernel_initializer=tf.initializers.random_uniform(-stdv, stdv))
    out_pos = tf.layers.dense(attended_pos, 512, activation=None, name=f'{name}/proj2',
                use_bias=False,
                kernel_initializer=tf.initializers.random_uniform(-stdv, stdv))


    out = tf.concat([out_dat, out_pos], -1)
    out = dropout_only_if_training(out, HPARAMS['residual_dropout'])
    return make_layer_norm(input + out, f'{torch_name}.layer_norm', f'{name}/layer_norm')

def make_dense_relu_dense(input, torch_name, torch_type, name):
    # TODO: use name
    stdv = 1 / np.sqrt(512)
    initializer = tf.initializers.random_uniform(-stdv, stdv)
    mul1b = tf.layers.dense(input, 1024, kernel_initializer=initializer, bias_initializer=initializer)
    mul1b = tf.nn.relu(mul1b)
    mul1b = dropout_only_if_training(mul1b, HPARAMS['relu_dropout'])
    stdv2 = 1 / np.sqrt(1024)
    initializer2 = tf.initializers.random_uniform(-stdv2, stdv2)
    mul2b = tf.layers.dense(mul1b, 512, kernel_initializer=initializer2, bias_initializer=initializer2)
    return mul2b

def make_ff(input, torch_name, name):
    input_dat, input_pos = tf.split(input, 2, axis=-1)
    out_dat = make_dense_relu_dense(input_dat, torch_name, 'c', name=f"{name}/dat")
    out_pos = make_dense_relu_dense(input_pos, torch_name, 'p', name=f"{name}pos")
    out = tf.concat([out_dat, out_pos], -1)
    out = dropout_only_if_training(out, HPARAMS['residual_dropout'])
    return make_layer_norm(input + out, f'{torch_name}.layer_norm', f'{name}/layer_norm')

def make_stacks(input, nonpad_ids, dim_flat, dim_padded, valid_mask, num_stacks):
    res = input
    for i in range(num_stacks):
        res = make_attention(res, nonpad_ids, dim_flat, dim_padded, valid_mask, f'encoder.attn_{i}', name=f'attn_{i}')
        res = make_ff(res, f'encoder.ff_{i}', name=f'ff_{i}')
    return res

def make_bert_projection(word_features_packed):
    stdv = 1 / np.sqrt(config.hidden_size)
    initializer = tf.initializers.random_uniform(-stdv, stdv)
    input_dat = tf.layers.dense(word_features_packed, 512, kernel_initializer=initializer, use_bias=False)
    return input_dat

def make_encoder(input_dat, nonpad_ids, dim_flat, dim_padded, valid_mask, num_stacks):
    input_pos_flat = tf.tile(position_table[:dim_padded[1]], [dim_padded[0], 1])
    input_pos = tf.gather(input_pos_flat, nonpad_ids)

    input_joint = tf.concat([input_dat, input_pos], -1)
    input_joint = make_layer_norm(input_joint, 'embedding.layer_norm', 'embedding/layer_norm')

    word_out = make_stacks(input_joint, nonpad_ids, dim_flat, dim_padded, valid_mask, num_stacks)
    return word_out

# %%

def get_word_features():
    # input_mask is over subwords, whereas valid_mask is over words
    sentence_lengths = tf.reduce_sum(word_end_mask, -1)
    valid_mask = (tf.range(tf.reduce_max(sentence_lengths))[None,:] < sentence_lengths[:, None])
    dim_padded = tf.shape(valid_mask)[:2]
    mask_flat = tf.reshape(valid_mask, (-1,))
    dim_flat = tf.shape(mask_flat)[:1]
    nonpad_ids = tf.to_int32(tf.where(mask_flat)[:,0])

    if not args.disable_bert:
        subword_features = model.get_sequence_output()
        word_features_packed = tf.gather(
            tf.reshape(subword_features, [-1, int(subword_features.shape[-1])]),
            tf.to_int32(tf.where(tf.reshape(word_end_mask, (-1,))))[:,0])

    if args.num_nonbert_layers > 0:
        if args.disable_bert:
            raise NotImplementedError("Using only factored self-attention without BERT is not implemented yet")
        else:
            print("Applying factored self-attention on top of BERT")
            input_dat = make_bert_projection(word_features_packed)

        word_features_packed = make_encoder(input_dat, nonpad_ids, dim_flat, dim_padded, valid_mask, num_stacks=args.num_nonbert_layers)

    # XXX(nikita): feature downscaling should be implemented in the c++ portion
    # of the network (not here)
    # The idea behind rescaling is that the code mixes BERT vectors and vectors
    # that are output by an LSTM, which would be a magnitude mismatch without
    # any rescaling.
    if args.feature_downscale != 1.0:
        print("RESCALING word features: dividing by {}".format(args.feature_downscale))
        word_features_packed = word_features_packed / args.feature_downscale

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

if not args.disable_bert:
    saver = tf.train.Saver(bert_tvars)
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
    """Sets up backward pass and optimizer, with support for subbatching"""
    global_step = tf.train.get_or_create_global_step()

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
    subbatch_grads = tf.gradients(ys=ys, xs=tvars, grad_ys=grad_ys)

    grad_accumulators = []
    accumulator_assignments = []
    accumulator_clears = []
    for grad, param in zip(subbatch_grads, tvars):
        if grad is None or param is None:
            grad_accumulators.append(None)
            continue

        param_name = bert.optimization.AdamWeightDecayOptimizer._get_variable_name(None, param.name)
        grad_accumulator = tf.get_variable(
            name=param_name + "/grad_accumulator",
            shape=param.shape.as_list(),
            dtype=tf.float32,
            trainable=False,
            initializer=tf.zeros_initializer())
        grad_accumulators.append(grad_accumulator)

        accumulator_assignments.append(tf.assign_add(grad_accumulator, grad))
        accumulator_clears.append(tf.assign(grad_accumulator, tf.zeros_like(grad_accumulator), use_locking=True))

    accumulate_op = tf.group(*accumulator_assignments, name='accumulate')
    zero_grad_op = tf.group(*accumulator_clears, name='zero_grad')

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

    bert_grads = [grad for (grad, tvar) in zip(grad_accumulators, tvars) if tvar in bert_tvars]
    extra_grads = [grad for (grad, tvar) in zip(grad_accumulators, tvars) if tvar not in bert_tvars]
    extra_tvars = [tvar for (grad, tvar) in zip(grad_accumulators, tvars) if tvar not in bert_tvars]
    if bert_grads and any([grad is not None for grad in bert_grads]):
        bert_grads_norm = conditional_print(
            do_print,
            tf.global_norm(bert_grads),
            "Gradient norm for BERT parameters: "
            )
        bert_grads, _ = tf.clip_by_global_norm(bert_grads, clip_norm=1.0, use_norm=bert_grads_norm)
    if extra_grads and any([grad is not None for grad in extra_grads]):
        extra_grads_norm = conditional_print(
            do_print,
            tf.global_norm(extra_grads),
            "Gradient norm for non-BERT encoder parameters: "
            )
        extra_grads, _ = tf.clip_by_global_norm(extra_grads, clip_norm=200.0, use_norm=extra_grads_norm)

    grads = list(bert_grads) + list(extra_grads)
    tvars = list(bert_tvars) + list(extra_tvars)

    train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)], name='train')

    return accumulate_op, train_op, zero_grad_op, learning_rate_var, warmup_steps_var

# %%

accumulate_op, train_op, zero_grad_op, learning_rate_var, warmup_steps_var = create_optimizer([word_features], [word_features_grad])

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
accumulate_op: {accumulate_op.name}
train_op: {train_op.name}
zero_grad_op: {zero_grad_op.name}
save_op: {saver.saver_def.save_tensor_name}
restore_op: {saver.saver_def.restore_op_name}
checkpoint_name: {saver.saver_def.filename_tensor_name}
""")

# %%

with open(args.bert_output_file, 'wb') as f:
    f.write(sess.graph_def.SerializeToString())

print("Saved tensorflow graph to to", args.bert_output_file)

# %%
