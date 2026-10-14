"""Token-by-token recurrence of the KDA forward pass, straight from the operator definition.

q, k, v, g : [1, T, H, 128] bf16     beta : [1, T, H] bf16 (logits)
A_log : [H] f32     dt_bias : [H*128] f32     initial_state : [N, H, 128(V), 128(K)] f32
cu_seqlens : [N+1] int64
Returns output [1, T, H, 128] bf16 and final_state [N, H, 128(V), 128(K)] f32.

    q, k <- L2-normalized per token and head        beta <- sigmoid(beta)
    log-decay a_t = lower_bound * sigmoid(exp(A_log) * (g_t + dt_bias))     (per key channel)
    S <- S * exp(a_t)  (scales key rows);  S <- S + beta_t * k_t (v_t - k_t^T S);  o_t = scale * q_t^T S
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def kda_reference(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, lower_bound=-5.0,
                  dtype=torch.float64):
    H, D = q.shape[2], q.shape[3]
    qn = F.normalize(q[0].to(dtype), dim=-1, eps=1e-6)
    kn = F.normalize(k[0].to(dtype), dim=-1, eps=1e-6)
    vf = v[0].to(dtype)
    a = lower_bound * torch.sigmoid(A_log.to(dtype).exp().view(H, 1) * (g[0].to(dtype) + dt_bias.to(dtype).view(H, D)))
    decay = a.exp()
    b = torch.sigmoid(beta[0].to(dtype))
    cu = cu_seqlens.tolist()
    out = torch.empty_like(vf)
    final = torch.empty(len(cu) - 1, H, D, D, dtype=torch.float32, device=q.device)
    for n in range(len(cu) - 1):
        S = initial_state[n].to(dtype).transpose(-1, -2).clone()  # [H, K, V]
        for t in range(cu[n], cu[n + 1]):
            S *= decay[t][..., None]
            kS = torch.einsum("hk,hkv->hv", kn[t], S)
            S += kn[t][..., None] * (b[t][:, None] * (vf[t] - kS))[:, None, :]
            out[t] = scale * torch.einsum("hk,hkv->hv", qn[t], S)
        final[n] = S.transpose(-1, -2).float()
    return out[None].to(torch.bfloat16), final
