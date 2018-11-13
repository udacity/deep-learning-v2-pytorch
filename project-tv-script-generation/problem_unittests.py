from unittest.mock import MagicMock, patch
import numpy as np
import torch


class _TestNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(_TestNN, self).__init__()
        self.decoder = torch.nn.Linear(input_size, output_size)
        self.forward_called = False
    
    def forward(self, nn_input, hidden):
        self.forward_called = True
        output = self.decoder(nn_input)
        
        return output, hidden


def _print_success_message():
    print('Tests Passed')


class AssertTest(object):
    def __init__(self, params):
        self.assert_param_message = '\n'.join([str(k) + ': ' + str(v) + '' for k, v in params.items()])
    
    def test(self, assert_condition, assert_message):
        assert assert_condition, assert_message + '\n\nUnit Test Function Parameters\n' + self.assert_param_message


def test_create_lookup_tables(create_lookup_tables):
    test_text = '''
        Moe_Szyslak Moe's Tavern Where the elite meet to drink
        Bart_Simpson Eh yeah hello is Mike there Last name Rotch
        Moe_Szyslak Hold on I'll check Mike Rotch Mike Rotch Hey has anybody seen Mike Rotch lately
        Moe_Szyslak Listen you little puke One of these days I'm gonna catch you and I'm gonna carve my name on your back with an ice pick
        Moe_Szyslak Whats the matter Homer You're not your normal effervescent self
        Homer_Simpson I got my problems Moe Give me another one
        Moe_Szyslak Homer hey you should not drink to forget your problems
        Barney_Gumble Yeah you should only drink to enhance your social skills'''
    
    test_text = test_text.lower()
    test_text = test_text.split()
    
    vocab_to_int, int_to_vocab = create_lookup_tables(test_text)
    
    # Check types
    assert isinstance(vocab_to_int, dict),\
        'vocab_to_int is not a dictionary.'
    assert isinstance(int_to_vocab, dict),\
        'int_to_vocab is not a dictionary.'
    
    # Compare lengths of dicts
    assert len(vocab_to_int) == len(int_to_vocab),\
        'Length of vocab_to_int and int_to_vocab don\'t match. ' \
        'vocab_to_int is length {}. int_to_vocab is length {}'.format(len(vocab_to_int), len(int_to_vocab))

    # Make sure the dicts have the same words
    vocab_to_int_word_set = set(vocab_to_int.keys())
    int_to_vocab_word_set = set(int_to_vocab.values())

    assert not (vocab_to_int_word_set - int_to_vocab_word_set),\
    'vocab_to_int and int_to_vocab don\'t have the same words.' \
        '{} found in vocab_to_int, but not in int_to_vocab'.format(vocab_to_int_word_set - int_to_vocab_word_set)
    assert not (int_to_vocab_word_set - vocab_to_int_word_set),\
        'vocab_to_int and int_to_vocab don\'t have the same words.' \
        '{} found in int_to_vocab, but not in vocab_to_int'.format(int_to_vocab_word_set - vocab_to_int_word_set)
    
    # Make sure the dicts have the same word ids
    vocab_to_int_word_id_set = set(vocab_to_int.values())
    int_to_vocab_word_id_set = set(int_to_vocab.keys())
    
    assert not (vocab_to_int_word_id_set - int_to_vocab_word_id_set),\
        'vocab_to_int and int_to_vocab don\'t contain the same word ids.' \
        '{} found in vocab_to_int, but not in int_to_vocab'.format(vocab_to_int_word_id_set - int_to_vocab_word_id_set)
    assert not (int_to_vocab_word_id_set - vocab_to_int_word_id_set),\
        'vocab_to_int and int_to_vocab don\'t contain the same word ids.' \
        '{} found in int_to_vocab, but not in vocab_to_int'.format(int_to_vocab_word_id_set - vocab_to_int_word_id_set)
    
    # Make sure the dicts make the same lookup
    missmatches = [(word, id, id, int_to_vocab[id]) for word, id in vocab_to_int.items() if int_to_vocab[id] != word]
    
    assert not missmatches,\
        'Found {} missmatche(s). First missmatch: vocab_to_int[{}] = {} and int_to_vocab[{}] = {}'.format(len(missmatches),
                                                                                                          *missmatches[0])
    
    assert len(vocab_to_int) > len(set(test_text))/2,\
        'The length of vocab seems too small.  Found a length of {}'.format(len(vocab_to_int))
    
    _print_success_message()


def test_tokenize(token_lookup):
    symbols = set(['.', ',', '"', ';', '!', '?', '(', ')', '-', '\n'])
    token_dict = token_lookup()
    
    # Check type
    assert isinstance(token_dict, dict), \
        'Returned type is {}.'.format(type(token_dict))

    # Check symbols
    missing_symbols = symbols - set(token_dict.keys())
    unknown_symbols = set(token_dict.keys()) - symbols

    assert not missing_symbols, \
    'Missing symbols: {}'.format(missing_symbols)
    assert not unknown_symbols, \
        'Unknown symbols: {}'.format(unknown_symbols)

    # Check values type
    bad_value_type = [type(val) for val in token_dict.values() if not isinstance(val, str)]
    
    assert not bad_value_type,\
        'Found token as {} type.'.format(bad_value_type[0])

    # Check for spaces
    key_has_spaces = [k for k in token_dict.keys() if ' ' in k]
    val_has_spaces = [val for val in token_dict.values() if ' ' in val]
    
    assert not key_has_spaces,\
        'The key "{}" includes spaces. Remove spaces from keys and values'.format(key_has_spaces[0])
    assert not val_has_spaces,\
    'The value "{}" includes spaces. Remove spaces from keys and values'.format(val_has_spaces[0])
    
    # Check for symbols in values
    symbol_val = ()
    for symbol in symbols:
        for val in token_dict.values():
            if symbol in val:
                symbol_val = (symbol, val)

    assert not symbol_val,\
    'Don\'t use a symbol that will be replaced in your tokens. Found the symbol {} in value {}'.format(*symbol_val)
    
    _print_success_message()


def test_rnn(RNN, train_on_gpu):
    batch_size = 50
    sequence_length = 3
    vocab_size = 20
    output_size=20
    embedding_dim=15
    hidden_dim = 10
    n_layers = 2
    
    # create test RNN
    # params: (vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    
    # create test input
    a = np.random.randint(vocab_size, size=(batch_size, sequence_length))
    #b = torch.LongTensor(a)
    b = torch.from_numpy(a)
    hidden = rnn.init_hidden(batch_size)
    
    
    if(train_on_gpu):
        rnn.cuda()
        b = b.cuda()
    
    output, hidden_out = rnn(b, hidden)
    
    assert_test = AssertTest({
                             'Input Size': vocab_size,
                             'Output Size': output_size,
                             'Hidden Dim': hidden_dim,
                             'N Layers': n_layers,
                             'Batch Size': batch_size,
                             'Sequence Length': sequence_length,
                             'Input': b})
    
    # initialization
    correct_hidden_size = (n_layers, batch_size, hidden_dim)
    assert_condition = hidden[0].size() == correct_hidden_size
    assert_message = 'Wrong hidden state size. Expected type {}. Got type {}'.format(correct_hidden_size, hidden[0].size())
    assert_test.test(assert_condition, assert_message)
    
    # output of rnn
    correct_hidden_size = (n_layers, batch_size, hidden_dim)
    assert_condition = hidden_out[0].size() == correct_hidden_size
    assert_message = 'Wrong hidden state size. Expected type {}. Got type {}'.format(correct_hidden_size, hidden_out[0].size())
    assert_test.test(assert_condition, assert_message)
    
    correct_output_size = (batch_size, output_size)
    assert_condition = output.size() == correct_output_size
    assert_message = 'Wrong output size. Expected type {}. Got type {}'.format(correct_output_size, output.size())
    assert_test.test(assert_condition, assert_message)
    
    _print_success_message()


def test_forward_back_prop(RNN, forward_back_prop, train_on_gpu):
    batch_size = 200
    input_size = 20
    output_size = 10
    sequence_length = 3
    embedding_dim=15
    hidden_dim = 10
    n_layers = 2
    learning_rate = 0.01
    
    # create test RNN
    rnn = RNN(input_size, output_size, embedding_dim, hidden_dim, n_layers)
    
    mock_decoder = MagicMock(wraps=_TestNN(input_size, output_size))
    if train_on_gpu:
        mock_decoder.cuda()
    
    mock_decoder_optimizer = MagicMock(wraps=torch.optim.Adam(mock_decoder.parameters(), lr=learning_rate))
    mock_criterion = MagicMock(wraps=torch.nn.CrossEntropyLoss())
    
    with patch.object(torch.autograd, 'backward', wraps=torch.autograd.backward) as mock_autograd_backward:
        inp = torch.FloatTensor(np.random.rand(batch_size, input_size))
        target = torch.LongTensor(np.random.randint(output_size, size=batch_size))
        
        hidden = rnn.init_hidden(batch_size)
        
        loss, hidden_out = forward_back_prop(mock_decoder, mock_decoder_optimizer, mock_criterion, inp, target, hidden)
    
    assert (hidden_out[0][0]==hidden[0][0]).sum()==batch_size*hidden_dim, 'Returned hidden state is the incorrect size.'
    
    assert mock_decoder.zero_grad.called or mock_decoder_optimizer.zero_grad.called, 'Didn\'t set the gradients to 0.'
    assert mock_decoder.forward_called, 'Forward propagation not called.'
    assert mock_autograd_backward.called, 'Backward propagation not called'
    assert mock_decoder_optimizer.step.called, 'Optimization step not performed'
    assert type(loss) == float, 'Wrong return type. Expected {}, got {}'.format(float, type(loss))
    
    _print_success_message()
