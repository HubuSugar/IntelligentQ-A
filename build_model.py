# Seq2Seq模型的编码层
def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            # 前向RNN
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_fw, input_keep_prob=keep_prob)

            # 后向RNN
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_bw, input_keep_prob=keep_prob)

            # 双向RNN
            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=rnn_inputs,
                                                                    sequence_length=sequence_length, dtype=tf.float32)
    # 因为使用的是双向rnn网络，所以要将两层的结果concat起来
    enc_output = tf.concat(enc_output, 2)
    return enc_output, enc_state


# 创建训练的decoder
def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer,
                            vocab_size, max_summary_length):
    # 训练层采用TrainingHelper函数                        
    training_helper = seq.TrainingHelper(inputs=dec_embed_input,
                                         sequence_length=summary_length,
                                         time_major=False)

    training_decoder = seq.BasicDecoder(cell=dec_cell,
                                        helper=training_helper,
                                        initial_state=initial_state,
                                        output_layer=output_layer)
    # train_dec_outputs, train_dec_last_state, _
    # 根据tensorflow1.2的版本改动，之前tensorflow1.1版本dynamic_decode返回的是两个参数，目前返回的是三个参数
    train_dec_outputs, train_dec_last_state, _ = seq.dynamic_decode(training_decoder,
                                                                    output_time_major=False,
                                                                    impute_finished=True,
                                                                    maximum_iterations=max_summary_length)
    return train_dec_outputs


# 创建测试的decoder
def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_summary_length, batch_size):
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    # 预测层采用GreedyEmbeddingHelper函数
    inference_helper = seq.GreedyEmbeddingHelper(embeddings,
                                                 start_tokens,
                                                 end_token)
    inference_decoder = seq.BasicDecoder(dec_cell,
                                         inference_helper,
                                         initial_state,
                                         output_layer)
    infer_dec_outputs, infer_dec_last_state, _ = seq.dynamic_decode(inference_decoder,
                                                                    output_time_major=False,
                                                                    impute_finished=True,
                                                                    maximum_iterations=max_summary_length)

    return infer_dec_outputs


# 创建真正的解码层 引入注意力机制
def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length,
                   max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob)
    # 全连接层
    output_layer = Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    attn_mech = seq.BahdanauAttention(rnn_size,
                                      enc_output,
                                      text_length,
                                      normalize=False,
                                      name='BahdanauAttention')

    dec_cell = seq.AttentionWrapper(cell=dec_cell,
                                    attention_mechanism=attn_mech,
                                    attention_layer_size=rnn_size,
                                    name='Attention_Wrapper')
    # tensorflow1.2版本变化
    initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    # 引入注意力机制
    # initial_state = seq.AttentionWrapperState(cell_state=enc_state[0],
    #                                           attention=_zero_state_tensors(rnn_size,
    #                                                                         batch_size,
    #                                                                         tf.float32),
    #                                           time=0,
    #                                           alignments=(),
    #                                           alignment_history=(),
    #                                           )

    with tf.variable_scope("decode"):
        train_dec_outputs = training_decoding_layer(dec_embed_input,
                                                     summary_length,
                                                     dec_cell,
                                                     initial_state,
                                                     output_layer,
                                                     vocab_size,
                                                     max_summary_length)
    with tf.variable_scope("decode", reuse=True):
        inference_dec_outputs = inference_decoding_layer(embeddings,
                                                         vocab_to_int['<GO>'],
                                                         vocab_to_int['<EOS>'],
                                                         dec_cell,
                                                         initial_state,
                                                         output_layer,
                                                         max_summary_length,
                                                         batch_size)
    return train_dec_outputs, inference_dec_outputs


# 创建序列模型
def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
    # 使用词向量矩阵作为词嵌入向量
    embeddings = word_embedding_matrix

    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)

    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)

    train_dec_outputs, inference_dec_outputs = decoding_layer(dec_embed_input,
                                                       embeddings,
                                                       enc_output,
                                                       enc_state,
                                                       vocab_size,
                                                       text_length,
                                                       summary_length,
                                                       max_summary_length,
                                                       rnn_size,
                                                       vocab_to_int,
                                                       keep_prob,
                                                       batch_size,
                                                       num_layers)

    return train_dec_outputs, inference_dec_outputs
