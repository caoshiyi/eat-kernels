"""Exact definitions of the two KDA operators of tasks/amd_kda_prod_mi350x (the production contract).

prefill: chunked-prefill KDA as vLLM's Kimi-K3 ROCm layer calls it (`chunk_kda_prefill` with the fused backend):
  inputs are the post-conv q / k / v, the raw gate and beta projections, a paged fp32 state pool updated in place,
  per-sequence `has_initial_state`, and an optional prefix-cache checkpoint per sequence written into the pool.
decode:  fused conv1d + KDA + gated RMSNorm decode / speculative verify step (vLLM / AITER conventions), unchanged
  from tasks/amd_kda_decode_mi350x.

Both operators share the state pool layout: fp32 [slots, H, 128(V), 128(K)] (value-major), slot 0 never used.
"""
import torch
import torch.nn.functional as F

D = 128
W = 4
CHUNK = 64  # checkpoint offsets are multiples of this (vLLM KDA_CHECKPOINT_ALIGNMENT)


def kda_gate(raw_g, A_log, dt_bias, lower_bound, dtype=torch.float32):
    """Log-space decay: lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias)). raw_g: [..., H, K]."""
    H, K = raw_g.shape[-2:]
    return lower_bound * torch.sigmoid(A_log.to(dtype).view(H, 1).exp() * (raw_g.to(dtype) + dt_bias.to(dtype).view(H, K)))


@torch.no_grad()
def kda_prefill_reference(q, k, v, raw_g, raw_beta, A_log, dt_bias, lower_bound, scale,
                          state_cache, state_indices, has_initial_state, cu_seqlens,
                          checkpoint_offsets=None, checkpoint_state_indices=None, dtype=torch.float32):
    """Token-by-token recurrence (all sequences advanced together; finished sequences are frozen exactly).

    q, k, v, raw_g [1, T, H, 128] bf16 (any strides); raw_beta [1, T, H] bf16 (any strides); A_log [H] f32;
    dt_bias [H*128] f32; state_cache [slots, H, 128(V), 128(K)] f32 (slot stride may exceed H*128*128);
    state_indices [N] int32; has_initial_state [N] bool; cu_seqlens [N+1] int32/int64;
    checkpoint_offsets [N] int32 or None; checkpoint_state_indices [N] int32 or None.

    Sequence n starts from state_cache[state_indices[n]] if has_initial_state[n] else from zeros (the slot is then
    not read: it may hold anything, including NaN). Its final state is written to state_cache[state_indices[n]].
    If checkpoint_offsets[n] = o > 0 and checkpoint_state_indices[n] = r >= 0, the state after the first o tokens
    of the sequence (o is a multiple of 64 and o <= len_n) is written to state_cache[r]. All written rows are
    distinct; no other row changes.

    Returns (output [1, T, H, 128] bf16, new_state_cache: a clone of state_cache with the writes applied).
    """
    dev = q.device
    T, H = q.shape[1], q.shape[2]
    cu = [int(x) for x in cu_seqlens.cpu().tolist()]
    N = len(cu) - 1
    lens = [cu[n + 1] - cu[n] for n in range(N)]
    L = max(lens) if lens else 0
    idx = state_indices.long().cpu().tolist()
    hinit = has_initial_state.bool().cpu().tolist()
    ck = {}
    if checkpoint_offsets is not None and checkpoint_state_indices is not None:
        for n, (o, r) in enumerate(zip(checkpoint_offsets.cpu().tolist(), checkpoint_state_indices.cpu().tolist())):
            if o > 0 and r >= 0:
                assert o % CHUNK == 0 and o <= lens[n], f"invalid checkpoint offset {o} for length {lens[n]}"
                ck.setdefault(o - 1, []).append((n, r))

    qn = F.normalize(q[0].to(dtype), dim=-1, eps=1e-6)
    kn = F.normalize(k[0].to(dtype), dim=-1, eps=1e-6)
    vf = v[0].to(dtype)
    gl = kda_gate(raw_g[0], A_log, dt_bias, lower_bound, dtype)
    bt = torch.sigmoid(raw_beta[0].to(dtype))

    # Pad to [N, L, ...]: padded steps have decay 1 and beta 0, so a finished sequence's state is unchanged exactly.
    pos = torch.full((N, L), T, dtype=torch.long)
    for n in range(N):
        pos[n, :lens[n]] = torch.arange(cu[n], cu[n + 1])
    pos = pos.to(dev)

    def pad(x, fill):
        x = torch.cat([x, torch.full((1, *x.shape[1:]), fill, dtype=x.dtype, device=dev)])
        return x[pos]

    Qp, Kp, Vp = pad(qn, 0.0), pad(kn, 0.0), pad(vf, 0.0)
    Gp, Bp = pad(gl, 0.0), pad(bt, 0.0)
    dec = Gp.exp()

    S = torch.zeros(N, H, D, D, dtype=dtype, device=dev)  # K-major [N, H, K, V]
    for n in range(N):
        if hinit[n]:
            S[n] = state_cache[idx[n]].to(dtype).transpose(-1, -2)
    out = torch.empty(N, L, H, D, dtype=dtype, device=dev)
    ck_states = {}
    for t in range(L):
        S = S * dec[:, t, :, :, None]
        kS = torch.einsum("nhk,nhkv->nhv", Kp[:, t], S)
        S = S + Kp[:, t, :, :, None] * (Bp[:, t, :, None] * (Vp[:, t] - kS))[:, :, None, :]
        out[:, t] = scale * torch.einsum("nhk,nhkv->nhv", Qp[:, t], S)
        for n, r in ck.get(t, ()):
            ck_states[r] = S[n].transpose(-1, -2).clone()

    o = torch.empty(T, H, D, dtype=dtype, device=dev)
    for n in range(N):
        o[cu[n]:cu[n + 1]] = out[n, :lens[n]]
    new_cache = state_cache.clone()
    for r, s in ck_states.items():
        new_cache[r] = s.to(new_cache.dtype)
    for n in range(N):
        new_cache[idx[n]] = S[n].transpose(-1, -2).to(new_cache.dtype)
    return o[None].to(torch.bfloat16), new_cache


@torch.no_grad()
def kda_decode_reference(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight,
                         ssm_state, state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound):
    """Fused decode / spec-verify step (identical to tasks/amd_kda_decode_mi350x/reference.py).

    One call processes B sequences with S tokens each (S = 1: plain decode; S = num_spec + 1: verify of S - 1 draft
    tokens), packed as T = B * S rows. Per sequence and token: causal depthwise conv1d (width 4) + SiLU over the
    packed q|k|v channels with a rolling bf16 conv cache; g = lower_bound * sigmoid(exp(A_log[h]) * (gate + dt_bias));
    beta = sigmoid(beta); q/k L2 norm (eps 1e-6 inside the sqrt), q *= 128 ** -0.5; the gated delta rule on the
    fp32 state S [V, K]; gated RMSNorm y = bf16(o) * rsqrt(mean(bf16(o)^2) + eps) * norm_weight * sigmoid(out_gate).

    plain decode (state_indices [B]): read and write ssm_state[state_indices[n]]; the conv cache
        conv_state[state_indices[n]] holds the last 3 inputs.
    spec verify (state_indices [B, S], num_accepted_tokens [B]): with a = num_accepted_tokens[n] (1..S) the step starts
        from ssm_state[state_indices[n, a - 1]] and writes the state after token t to ssm_state[state_indices[n, t]]
        (slots <= 0 are never written). The conv cache conv_state[state_indices[n, 0]] has 2 + S taps: the history
        is taps [a - 1, a + 2); afterwards taps [0, 2 + S) hold the last 2 history taps followed by the S inputs.

    Returns (out [T, H*128] bf16, new conv_state, new ssm_state); the inputs are not modified.
    """
    T, C = mixed_qkv.shape
    dim = C // 3
    H = dim // D
    B = cu_seqlens.numel() - 1
    S = T // B
    assert T == B * S and torch.equal(cu_seqlens.cpu(), torch.arange(0, T + 1, S, dtype=cu_seqlens.dtype))
    spec = num_accepted_tokens is not None
    dev = mixed_qkv.device
    conv_new, ssm_new = conv_state.clone(), ssm_state.clone()
    ar = torch.arange(B, device=dev)

    if spec:
        acc = num_accepted_tokens.long()
        read_slot = state_indices[ar, acc - 1].long()
        cslot = state_indices[:, 0].long()
        start = acc - 1
    else:
        read_slot = state_indices.long()
        cslot = read_slot
        start = torch.zeros(B, dtype=torch.long, device=dev)
    taps = start[:, None] + torch.arange(W - 1, device=dev)[None, :]
    hist = conv_state[cslot[:, None], taps]
    x = mixed_qkv.view(B, S, C)
    window = torch.cat([hist, x], dim=1)
    w = conv_weight.float().permute(1, 0, 2).reshape(W, C)

    st = ssm_state[read_slot].float()
    A = torch.exp(A_log.float())[None, :, None]
    dtb = dt_bias.float().view(1, H, D)
    nw = norm_weight.float()
    out = torch.empty(T, dim, dtype=torch.bfloat16, device=dev)
    for t in range(S):
        c = sum(window[:, t + j].float() * w[j] for j in range(W))
        c = c * torch.sigmoid(c)
        q, k, v = c[:, :dim].view(B, H, D), c[:, dim:2 * dim].view(B, H, D), c[:, 2 * dim:].view(B, H, D)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * D ** -0.5
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        tok = ar * S + t
        g = lower_bound * torch.sigmoid(A * (gate[tok].float() + dtb))
        b = torch.sigmoid(beta[tok].float())
        st = st * torch.exp(g)[:, :, None, :]
        u = b[..., None] * (v - (st * k[:, :, None, :]).sum(-1))
        st = st + u[..., None] * k[:, :, None, :]
        o = (st * q[:, :, None, :]).sum(-1)
        if spec:
            ws = state_indices[:, t].long()
            keep = ws > 0
            ssm_new[ws[keep]] = st[keep]
        ob = o.bfloat16().float()
        rstd = torch.rsqrt((ob * ob).mean(-1, keepdim=True) + norm_eps)
        og = out_gate[tok].float().view(B, H, D)
        out[tok] = (ob * rstd * nw * torch.sigmoid(og)).bfloat16().view(B, dim)
    if spec:
        conv_new[cslot, :W - 2 + S] = window[:, 1:]
    else:
        ssm_new[read_slot] = st
        conv_new[cslot] = window[:, -(W - 1):]
    return out, conv_new, ssm_new
