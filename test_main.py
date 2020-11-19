from torch import optim
optimizer_parameters = [p for p in encoder.parameters() if p.requires_grad ] + [p for p in decoder.parameters() if p.requires_grad ]
meta_optim = optim.Adam(optimizer_parameters, lr=0.01)

encoder_named_parameters = [ (n,p) for n,p in encoder.named_parameters() ]
decoder_named_parameters = [ (n,p) for n,p in decoder.named_parameters() ] + [ ('pred_layer.proj.weight',decoder.pred_layer.proj.weight)] # bug fix
encoder_fast_params = construct_fast_params(encoder_named_parameters)
decoder_fast_params = construct_fast_params(decoder_named_parameters)
encoder_named_params = [ n for n,p in encoder_named_parameters ]
decoder_named_params = [ n for n,p in decoder_named_parameters ]

update_rate = 0.01
inner_loop = 5

losses = []
for i in range(inner_loop):
    enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False, params=encoder_fast_params)
    enc1 = enc1.transpose(0, 1)
    dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1,
                   params=decoder_fast_params)
    _, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False,
                      params=decoder_fast_params['pred_layer']['proj'])

    encoder_parameters = deconstruct_fast_params(encoder_fast_params, encoder_named_params)
    decoder_parameters = deconstruct_fast_params(decoder_fast_params, decoder_named_params)

    encoder_named_parameters = [(n, p) for n, p in zip(encoder_named_params, encoder_parameters)]
    decoder_named_parameters = [(n, p) for n, p in zip(decoder_named_params, decoder_parameters)]

    encoder_grads = grad(loss, encoder_parameters, allow_unused=True, retain_graph=True)
    decoder_grads = grad(loss, decoder_parameters)

    encoder_fast_params = new_fast_params(encoder_named_parameters, encoder_grads)
    decoder_fast_params = new_fast_params(decoder_named_parameters, decoder_grads)

    enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False, params=encoder_fast_params)
    enc1 = enc1.transpose(0, 1)
    dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1,
                   params=decoder_fast_params)
    _, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False,
                      params=decoder_fast_params['pred_layer']['proj'])
    if i == inner_loop -1:
        losses.append(loss)

meta_optim.zero_grad()
loss = sum(losses)
loss.backward()
meta_optim.step()